"""FastAPI Application Entrypoint for Agentic AI RAG Chatbot."""

import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.config import settings
from app.ingestion import process_pdf_into_chunks
from app.embeddings import EmbeddingService
from app.vectorstore import ChromaStore
from app.graph import RAGGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description="Simple, clean, grounded RAG Chatbot API using LangGraph, ChromaDB, Gemini, and FastAPI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_vectorstore = ChromaStore()
_rag_graph = RAGGraph(vectorstore=_vectorstore)


class HealthResponse(BaseModel):
    """Schema for health endpoint response."""

    status: str = Field("healthy", description="Status string.")
    service: str = Field("Agentic AI RAG Chatbot API", description="Service name.")
    version: str = Field("1.0.0", description="API version.")
    total_indexed_chunks: int = Field(..., description="ChromaDB indexed chunk count.")


class UploadResponse(BaseModel):
    """Schema for PDF upload response."""

    filename: str = Field(..., description="Uploaded PDF filename.")
    indexed_chunks: int = Field(..., description="Number of text chunks indexed into ChromaDB.")


class ChatRequest(BaseModel):
    """Schema for chat request payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question regarding the Agentic AI eBook.",
        json_schema_extra={"example": "What is Agentic AI?"},
    )


class RetrievedContextChunk(BaseModel):
    """Schema for individual retrieved context chunk."""

    text: str = Field(..., description="Chunk text content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary.")
    score: float = Field(..., description="ChromaDB distance/similarity score.")


class ChatResponse(BaseModel):
    """Schema for chat response payload."""

    answer: str = Field(..., description="Grounded answer or refusal message.")
    retrieved_context: List[RetrievedContextChunk] = Field(..., description="Retrieved context chunks.")
    confidence: float = Field(..., description="Calculated retrieval confidence score.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serves the interactive Web UI for PDF upload and Chat Q&A."""
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Agentic AI RAG Chatbot API is running. Visit <a href='/docs'>/docs</a> for Swagger UI.</h1>"


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health():
    """Health check endpoint returning service status and indexed chunks count."""
    try:
        count = _vectorstore.get_count()
        return HealthResponse(
            status="healthy",
            service="Agentic AI RAG Chatbot API",
            version="1.0.0",
            total_indexed_chunks=count,
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service unavailable.",
        )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file, processes page text into chunks, and updates ChromaDB vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for upload.",
        )

    try:
        save_dir = os.path.dirname(settings.PDF_PATH)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file.filename)

        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        logger.info(f"Uploaded PDF saved to '{save_path}' ({len(content)} bytes).")

        # Process PDF into text chunks
        chunks = process_pdf_into_chunks(pdf_path=save_path)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract any readable text chunks from the uploaded PDF.",
            )

        # Generate embeddings and update vector store
        embedder = EmbeddingService()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_documents(texts)

        _vectorstore.reset_collection()
        _vectorstore.add_chunks(chunks, embeddings)

        return UploadResponse(
            filename=file.filename,
            indexed_chunks=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing the uploaded PDF: {str(e)}",
        )


@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """Chat endpoint submitting query to LangGraph RAG pipeline."""
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question text cannot be empty or whitespace.",
        )

    try:
        logger.info(f"Received API chat question: '{cleaned_question}'")
        res_state = _rag_graph.run(cleaned_question)

        formatted_context: List[RetrievedContextChunk] = []
        for doc in res_state.get("retrieved_documents", []):
            formatted_context.append(
                RetrievedContextChunk(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    score=round(doc.get("distance", 0.0), 4),
                )
            )

        return ChatResponse(
            answer=res_state.get("answer", ""),
            retrieved_context=formatted_context,
            confidence=round(res_state.get("confidence", 0.0), 4),
        )

    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing query.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
