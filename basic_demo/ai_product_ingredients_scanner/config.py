from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

    EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip()
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash").strip()
    VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash").strip()

    OUTPUT_DIMENSIONALITY = int(os.getenv("OUTPUT_DIMENSIONALITY", "768"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

    DATA_DIR = BASE_DIR / "data_pdfs"
    UPLOAD_DIR = BASE_DIR / "uploads"
    VECTOR_DIR = BASE_DIR / "vectorstore"
    INDEX_PATH = VECTOR_DIR / "faiss.index"
    META_PATH = VECTOR_DIR / "chunks.json"

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
