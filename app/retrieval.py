"""Retriever Module: Multi-Query Expansion, Hybrid (Dense + BM25) Search, RRF Fusion, and Reranking."""

import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.config import settings
from app.embeddings import EmbeddingService
from app.vectorstore import ChromaStore

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 matching."""
    return re.findall(r"\w+", text.lower())


def generate_query_variations(query: str) -> List[str]:
    """Generate multi-query variations to enhance retrieval recall."""
    variations = [query.strip()]
    q_lower = query.lower()

    if "agent" in q_lower and "agentic" not in q_lower:
        variations.append(query + " Agentic AI patterns")
    if "rag" in q_lower or "retrieval" in q_lower:
        variations.append(query + " Vector search knowledge base")
    if not any(q_lower.startswith(prefix) for prefix in ["what is", "how", "why", "describe"]):
        variations.append(f"What is {query} in Agentic AI architecture?")

    unique = []
    for v in variations:
        if v and v not in unique:
            unique.append(v)
    return unique[:3]


class BM25Retriever:
    """Sparse Lexical Retriever using rank_bm25 BM25Okapi."""

    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        if documents:
            corpus = [tokenize(doc["text"]) for doc in documents]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k documents based on BM25 lexical overlap."""
        if not self.bm25 or not self.documents or not query.strip():
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = dict(self.documents[idx])
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Combine dense vector and sparse lexical results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc.get("chunk_id") or doc.get("text", "")[:50]
        doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    for rank, doc in enumerate(sparse_results, start=1):
        chunk_id = doc.get("chunk_id") or doc.get("text", "")[:50]
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused = []
    for cid in sorted_ids:
        item = dict(doc_map[cid])
        item["rrf_score"] = round(rrf_scores[cid], 5)
        fused.append(item)
    return fused


def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks using lexical overlap and distance bonuses."""
    if not chunks:
        return []

    q_tokens = set(tokenize(query))
    scored_chunks = []

    for idx, doc in enumerate(chunks):
        doc_tokens = set(tokenize(doc.get("text", "")))
        overlap = len(q_tokens.intersection(doc_tokens))
        rrf_score = doc.get("rrf_score", 0.0)
        dist_bonus = 1.0 - min(1.0, doc.get("distance", 0.5))

        rank_score = (rrf_score * 10.0) + (overlap * 0.5) + (dist_bonus * 2.0) - (idx * 0.01)

        chunk_copy = dict(doc)
        chunk_copy["rerank_score"] = round(rank_score, 4)
        scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_n]


class Retriever:
    """Executes hybrid (Dense + Sparse BM25 + Multi-Query + RRF) retrieval against ChromaDB."""

    def __init__(
        self,
        vectorstore: Optional[ChromaStore] = None,
        embedder: Optional[EmbeddingService] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        self.vectorstore = vectorstore or ChromaStore()
        self.embedder = embedder or EmbeddingService()
        self.top_k = top_k or settings.TOP_K
        self.threshold = threshold if threshold is not None else settings.RELEVANCE_THRESHOLD
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._init_bm25()

    def _init_bm25(self):
        """Initialize sparse BM25 retriever from stored ChromaDB documents."""
        try:
            all_docs = self.vectorstore.get_all_documents()
            if all_docs:
                self.bm25_retriever = BM25Retriever(all_docs)
        except Exception as e:
            logger.warning(f"Could not initialize BM25 index: {e}")

    def calculate_confidence(self, distances: List[float]) -> float:
        """Calculate normalized confidence score from ChromaDB cosine distance d in [0, 2]."""
        if not distances:
            return 0.0
        min_dist = min(distances)
        confidence = max(0.0, 1.0 - min_dist)
        return round(confidence, 4)

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Perform hybrid retrieval combining multi-query expansion, dense search, BM25, and RRF fusion."""
        if not query or not query.strip():
            return {
                "retrieved_documents": [],
                "context": "",
                "retrieval_scores": [],
                "confidence": 0.0,
                "is_relevant": False,
                "query_variations": [],
            }

        logger.info(f"Executing hybrid retrieval for query: '{query[:60]}...'")
        variations = generate_query_variations(query)

        # 1. Multi-Query Dense Search
        all_dense = []
        seen_ids = set()
        for q in variations:
            q_embed = self.embedder.embed_query(q)
            dense_docs = self.vectorstore.query(q_embed, top_k=self.top_k)
            for doc in dense_docs:
                cid = doc.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_dense.append(doc)

        # 2. Sparse BM25 Search
        sparse_docs = []
        if self.bm25_retriever is None:
            self._init_bm25()
        if self.bm25_retriever:
            sparse_docs = self.bm25_retriever.retrieve(query, top_k=self.top_k)

        # 3. Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion(all_dense, sparse_docs)

        # 4. Context Reranking
        reranked_docs = rerank_chunks(query, fused_docs, top_n=self.top_k)

        if not reranked_docs:
            return {
                "retrieved_documents": [],
                "context": "",
                "retrieval_scores": [],
                "confidence": 0.0,
                "is_relevant": False,
                "query_variations": variations,
            }

        distances = [doc.get("distance", 0.5) for doc in reranked_docs if "distance" in doc]
        if not distances:
            distances = [0.5]
        confidence = self.calculate_confidence(distances)
        is_relevant = confidence >= self.threshold

        context_blocks = []
        for doc in reranked_docs:
            page = doc.get("page", "Unknown")
            context_blocks.append(f"[Page {page}]\n{doc['text']}")

        formatted_context = "\n\n---\n\n".join(context_blocks)

        return {
            "retrieved_documents": reranked_docs,
            "context": formatted_context,
            "retrieval_scores": distances,
            "confidence": confidence,
            "is_relevant": is_relevant,
            "query_variations": variations,
        }
