"""PDF Ingestion Module: Download, Extract PyMuPDF text, and Chunk."""

import os
import re
import logging
from typing import List, Dict, Any
import requests
import pymupdf  # PyMuPDF
from app.config import settings

logger = logging.getLogger(__name__)


def download_pdf(pdf_url: str = settings.PDF_URL, save_path: str = settings.PDF_PATH) -> str:
    """Download PDF file if not already cached locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        logger.info(f"PDF already exists at '{save_path}'.")
        return save_path

    logger.info(f"Downloading PDF from '{pdf_url}'...")
    res = requests.get(pdf_url, timeout=30)
    res.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(res.content)
    logger.info(f"PDF downloaded ({len(res.content)} bytes).")
    return save_path


def clean_text(text: str) -> str:
    """Clean extracted text by stripping control chars and normalizing whitespace."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str = settings.PDF_PATH) -> List[Dict[str, Any]]:
    """Extract page text preserving page numbers."""
    if not os.path.exists(pdf_path):
        pdf_path = download_pdf()

    doc = pymupdf.open(pdf_path)
    pages = []
    source = os.path.basename(pdf_path)

    for idx, page in enumerate(doc):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append({"page": idx + 1, "text": text, "source": source})

    doc.close()
    logger.info(f"Extracted {len(pages)} non-empty pages from PDF.")
    return pages


def split_text_recursively(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Recursive character splitting algorithm."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def _split(txt: str, seps: List[str]):
        if len(txt) <= chunk_size or not seps:
            if txt.strip():
                chunks.append(txt.strip())
            return

        sep = seps[0]
        next_seps = seps[1:]

        if sep == "":
            for i in range(0, len(txt), chunk_size - overlap):
                sub = txt[i : i + chunk_size].strip()
                if sub:
                    chunks.append(sub)
            return

        parts = txt.split(sep)
        curr = ""
        for p in parts:
            item = p if not curr else sep + p
            if len(item) > chunk_size:
                if curr.strip():
                    chunks.append(curr.strip())
                    curr = ""
                _split(p, next_seps)
            elif len(curr) + len(item) <= chunk_size:
                curr += item
            else:
                if curr.strip():
                    chunks.append(curr.strip())
                overlap_txt = curr[-overlap:] if len(curr) > overlap else ""
                curr = overlap_txt + item

        if curr.strip():
            chunks.append(curr.strip())

    _split(text, separators)
    return chunks


def process_pdf_into_chunks(
    pdf_path: str = settings.PDF_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Process PDF pages into chunks with metadata."""
    pages = extract_pages(pdf_path)
    all_chunks = []
    chunk_counter = 0

    for p in pages:
        split_texts = split_text_recursively(p["text"], chunk_size, chunk_overlap)
        for txt in split_texts:
            chunk_counter += 1
            cid = f"chunk_{chunk_counter:04d}"
            all_chunks.append(
                {
                    "chunk_id": cid,
                    "text": txt,
                    "metadata": {
                        "source": p["source"],
                        "page": p["page"],
                        "chunk_id": cid,
                    },
                }
            )

    logger.info(f"Generated {len(all_chunks)} chunks from PDF.")
    return all_chunks
