"""Application configuration settings using Pydantic Settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings class reading from environment variables."""

    # Gemini LLM Configuration
    GEMINI_API_KEY: Optional[str] = "mock_key_for_local_testing"
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    # ChromaDB Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "agentic_ai_knowledge"

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # RAG Retrieval Parameters
    TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.50

    # PDF Source Location
    PDF_URL: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    PDF_PATH: str = "data/Agentic-AI.pdf"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
