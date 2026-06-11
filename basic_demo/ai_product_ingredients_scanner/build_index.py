from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from config import Config
from services.gemini_service import GeminiService
from services.pdf_ingest import load_all_documents
from services.vector_store import save_store

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS index from PDF documents.")
    parser.add_argument("--pdf-dir", type=Path, default=Config.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Config.VECTOR_DIR)
    parser.add_argument("--chunk-size", type=int, default=Config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=Config.CHUNK_OVERLAP)
    parser.add_argument("--output-dim", type=int, default=Config.OUTPUT_DIMENSIONALITY)
    args = parser.parse_args()

    gemini = GeminiService(
        api_key=Config.GEMINI_API_KEY,
        embed_model=Config.EMBED_MODEL,
        chat_model=Config.CHAT_MODEL,
        vision_model=Config.VISION_MODEL,
        output_dimensionality=args.output_dim,
    )

    if not gemini.ready:
        raise SystemExit("Set GEMINI_API_KEY in .env before building the index.")

    docs = load_all_documents(args.pdf_dir, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    if not docs:
        raise SystemExit(f"No PDF documents found in: {args.pdf_dir}")

    texts = [d.text for d in docs]
    vectors = gemini.embed_texts(texts)
    chunks = [
        {
            "text": d.text,
            "source_pdf": d.source_pdf,
            "page_number": d.page_number,
            "chunk_index": d.chunk_index,
        }
        for d in docs
    ]

    save_store(args.out_dir / "faiss.index", args.out_dir / "chunks.json", vectors, chunks)
    print(f"Built index with {len(chunks)} chunks from {len(set(d.source_pdf for d in docs))} PDFs.")


if __name__ == "__main__":
    main()
