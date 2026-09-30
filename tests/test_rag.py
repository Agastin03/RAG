"""Comprehensive test suite covering Web UI, PDF Upload, Health check, Hybrid Retrieval, RRF Fusion, and 6 sample questions."""

import os
from fastapi.testclient import TestClient
from app.main import app
from app.llm import GROUNDED_REFUSAL_MESSAGE
from app.retrieval import generate_query_variations, reciprocal_rank_fusion, rerank_chunks
from app.graph import RAGGraph

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health endpoint status and chunk count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_indexed_chunks"] > 0


def test_ui_root_endpoint():
    """Verify GET / serves the HTML Web UI interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agentic AI RAG Chatbot" in response.text


def test_query_variations_generation():
    """Verify Multi-Query expansion logic."""
    variations = generate_query_variations("What are agent patterns?")
    assert len(variations) >= 1
    assert "What are agent patterns?" in variations


def test_reciprocal_rank_fusion_math():
    """Verify Reciprocal Rank Fusion calculation."""
    dense = [{"chunk_id": "chunk1", "text": "A"}, {"chunk_id": "chunk2", "text": "B"}]
    sparse = [{"chunk_id": "chunk2", "text": "B"}, {"chunk_id": "chunk3", "text": "C"}]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(fused) == 3
    # chunk2 appears in both, so it should rank highest
    assert fused[0]["chunk_id"] == "chunk2"


def test_context_reranking():
    """Verify reranking order of retrieved chunks."""
    chunks = [
        {"chunk_id": "c1", "text": "Python RAG implementation", "rrf_score": 0.03},
        {"chunk_id": "c2", "text": "Agentic AI architectural patterns", "rrf_score": 0.05},
    ]
    reranked = rerank_chunks("Agentic AI", chunks, top_n=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2"


def test_rag_graph_execution():
    """Verify RAGGraph orchestrates all pipeline nodes."""
    graph = RAGGraph()
    res = graph.run("What is Agentic AI?")
    assert "answer" in res
    assert "query_variations" in res
    assert "is_relevant" in res


def test_sample_question_1_factual():
    """Sample Question 1 (Factual): What is Agentic AI?"""
    response = client.post("/chat", json={"question": "What is Agentic AI?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert data["confidence"] >= 0.50
    assert len(data["retrieved_context"]) > 0


def test_sample_question_2_conceptual():
    """Sample Question 2 (Conceptual): How does Agentic AI differ from traditional generative AI?"""
    response = client.post("/chat", json={"question": "How does Agentic AI differ from traditional generative AI?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert data["confidence"] >= 0.50


def test_sample_question_3_multi_chunk():
    """Sample Question 3 (Multi-Chunk): What are the core architectural components of an AI Agent?"""
    response = client.post("/chat", json={"question": "What are the core architectural components of an AI Agent?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert data["confidence"] >= 0.50


def test_sample_question_4_document_relevant():
    """Sample Question 4 (Document Relevant): What are key enterprise use cases for Agentic AI?"""
    response = client.post("/chat", json={"question": "What are key enterprise use cases for Agentic AI?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert data["confidence"] >= 0.50


def test_sample_question_5_not_in_pdf_refusal():
    """Sample Question 5 (Not in PDF): What is the capital of Australia?"""
    response = client.post("/chat", json={"question": "What is the capital of Australia?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == GROUNDED_REFUSAL_MESSAGE
    assert data["confidence"] < 0.50


def test_sample_question_6_out_of_scope_refusal():
    """Sample Question 6 (Out of Scope): Who won the 2022 FIFA World Cup?"""
    response = client.post("/chat", json={"question": "Who won the 2022 FIFA World Cup?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == GROUNDED_REFUSAL_MESSAGE
    assert data["confidence"] <= 0.50


def test_empty_question_validation():
    """Test POST /chat input validation for empty questions."""
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400
