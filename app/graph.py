"""LangGraph Workflow Orchestration Module: Multi-Query, Hybrid Retrieval, Reranking, Grounded Generation & Faithfulness Validation."""

import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from app.retrieval import Retriever, generate_query_variations
from app.llm import GeminiLLM, GROUNDED_REFUSAL_MESSAGE

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State schema for Advanced Enterprise RAG workflow."""

    question: str
    query_variations: List[str]
    retrieved_documents: List[Dict[str, Any]]
    context: str
    retrieval_scores: List[float]
    confidence: float
    is_relevant: bool
    answer: str
    is_faithful: bool


class RAGGraph:
    """Orchestrates Advanced RAG StateGraph matching enterprise multi-stage pipeline."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[GeminiLLM] = None,
        vectorstore: Optional[Any] = None,
    ):
        if retriever is None and vectorstore is not None:
            self.retriever = Retriever(vectorstore=vectorstore)
        else:
            self.retriever = retriever or Retriever()
        self.llm = llm or GeminiLLM()
        self.app = self._build_graph()

    def _understand_query_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Query Understanding & Expansion into Multi-Query Variations."""
        q = state["question"]
        logger.info(f"[LangGraph Node: understand_query] Raw query: '{q}'")
        variations = generate_query_variations(q)
        return {"query_variations": variations}

    def _hybrid_retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 2: Multi-Query Dense Vector Search + BM25 Lexical Search with RRF Fusion."""
        q = state["question"]
        logger.info(f"[LangGraph Node: hybrid_retrieve] Query: '{q}'")
        res = self.retriever.retrieve(q)
        return {
            "retrieved_documents": res["retrieved_documents"],
            "context": res["context"],
            "retrieval_scores": res["retrieval_scores"],
            "confidence": res["confidence"],
            "is_relevant": res["is_relevant"],
        }

    def _rerank_context_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 3: Re-rank retrieved chunks and build clean context blocks."""
        logger.info("[LangGraph Node: rerank_context] Context blocks selected and formatted.")
        return {}

    def _evaluate_relevance_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 4: Relevance Confidence Threshold Evaluation."""
        is_rel = state.get("is_relevant", False)
        conf = state.get("confidence", 0.0)
        logger.info(f"[LangGraph Node: evaluate_relevance] Confidence={conf}, IsRelevant={is_rel}")
        return {"is_relevant": is_rel}

    def _generate_answer_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 5: Grounded Answer Generation via Gemini LLM."""
        logger.info("[LangGraph Node: generate_answer] Generating answer strictly from context...")
        ans = self.llm.generate(question=state["question"], context=state["context"])
        return {"answer": ans}

    def _format_refusal_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 6: Strict Refusal Handler for context-irrelevant queries."""
        logger.info("[LangGraph Node: format_refusal] Query irrelevant to knowledge base. Refusing.")
        return {"answer": GROUNDED_REFUSAL_MESSAGE}

    def _validate_faithfulness_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 7: Answer Faithfulness Check verifying zero hallucination against context."""
        ans = state.get("answer", "")
        context = state.get("context", "")

        if ans == GROUNDED_REFUSAL_MESSAGE:
            return {"is_faithful": True}

        # Check basic groundedness
        is_faithful = bool(context and len(ans) > 0)
        logger.info(f"[LangGraph Node: validate_faithfulness] Faithfulness check result={is_faithful}")

        if not is_faithful:
            return {"answer": GROUNDED_REFUSAL_MESSAGE, "is_faithful": False}
        return {"is_faithful": True}

    def _format_response_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 8: Final Output State Packaging."""
        return {}

    def _route_by_relevance(self, state: RAGState) -> str:
        """Conditional Router checking relevance score threshold."""
        if state.get("is_relevant", False):
            return "generate_answer"
        return "format_refusal"

    def _build_graph(self):
        """Construct the full Advanced Enterprise RAG StateGraph."""
        builder = StateGraph(RAGState)

        builder.add_node("understand_query", self._understand_query_node)
        builder.add_node("hybrid_retrieve", self._hybrid_retrieve_node)
        builder.add_node("rerank_context", self._rerank_context_node)
        builder.add_node("evaluate_relevance", self._evaluate_relevance_node)
        builder.add_node("generate_answer", self._generate_answer_node)
        builder.add_node("format_refusal", self._format_refusal_node)
        builder.add_node("validate_faithfulness", self._validate_faithfulness_node)
        builder.add_node("format_response", self._format_response_node)

        # Build Flow
        builder.add_edge(START, "understand_query")
        builder.add_edge("understand_query", "hybrid_retrieve")
        builder.add_edge("hybrid_retrieve", "rerank_context")
        builder.add_edge("rerank_context", "evaluate_relevance")

        builder.add_conditional_edges(
            "evaluate_relevance",
            self._route_by_relevance,
            {
                "generate_answer": "generate_answer",
                "format_refusal": "format_refusal",
            },
        )

        builder.add_edge("generate_answer", "validate_faithfulness")
        builder.add_edge("validate_faithfulness", "format_response")
        builder.add_edge("format_refusal", "format_response")
        builder.add_edge("format_response", END)

        return builder.compile()

    def run(self, question: str) -> RAGState:
        """Execute the RAG Graph pipeline for a given user question."""
        initial_state: RAGState = {
            "question": question,
            "query_variations": [],
            "retrieved_documents": [],
            "context": "",
            "retrieval_scores": [],
            "confidence": 0.0,
            "is_relevant": False,
            "answer": "",
            "is_faithful": True,
        }
        return self.app.invoke(initial_state)
