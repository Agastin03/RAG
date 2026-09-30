"""PDF Ingestion Script."""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.ingestion import process_pdf_into_chunks
from app.embeddings import EmbeddingService
from app.vectorstore import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion():
    """Execute reproducible PDF ingestion into ChromaDB."""
    logger.info("=== Starting Ingestion Process ===")

    chunks = process_pdf_into_chunks(pdf_path=settings.PDF_PATH)
    if not chunks:
        logger.error("No chunks extracted. Aborting.")
        return

    logger.info("Generating vector embeddings...")
    embedder = EmbeddingService()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    vstore = ChromaStore()
    if vstore.get_count() > 0:
        logger.info("Resetting existing collection for clean ingestion...")
        vstore.reset_collection()

    logger.info("Upserting chunks into ChromaDB...")
    vstore.add_chunks(chunks, embeddings)

    logger.info(f"=== Ingestion Complete: {vstore.get_count()} chunks indexed. ===")


if __name__ == "__main__":
    run_ingestion()
