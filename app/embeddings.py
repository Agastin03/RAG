"""Text Embedding Service Module."""

import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embedding vectors for document chunks and user queries."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Initializing EmbeddingService with model '{self.model_name}'...")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single user question."""
        if not text or not text.strip():
            return []
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding
