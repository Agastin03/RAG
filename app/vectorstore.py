"""ChromaDB Vector Store Management Module."""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from app.config import settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages persistent ChromaDB vector store initialization, indexing, and retrieval."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready at '{self.persist_directory}' "
            f"with {self.collection.count()} items."
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add chunks, embeddings, and metadata to ChromaDB."""
        if not chunks or not embeddings:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(ids)} chunks into collection '{self.collection_name}'.")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search in ChromaDB and return documents with metadata and distance."""
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return formatted

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for text, meta, dist in zip(docs, metas, dists):
            formatted.append(
                {
                    "text": text,
                    "metadata": meta,
                    "page": meta.get("page"),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id"),
                    "distance": float(dist),
                }
            )
        return formatted

    def get_count(self) -> int:
        """Return total document count in the collection."""
        return self.collection.count()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all stored documents and metadata for sparse BM25 indexing."""
        count = self.get_count()
        if count == 0:
            return []
        
        results = self.collection.get(include=["documents", "metadatas"])
        if not results or not results.get("documents"):
            return []

        formatted = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for chunk_id, text, meta in zip(ids, docs, metas):
            formatted.append(
                {
                    "text": text,
                    "metadata": meta or {},
                    "page": (meta or {}).get("page"),
                    "source": (meta or {}).get("source"),
                    "chunk_id": chunk_id,
                }
            )
        return formatted

    def reset_collection(self) -> None:
        """Reset the collection for fresh re-ingestion."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Reset collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
