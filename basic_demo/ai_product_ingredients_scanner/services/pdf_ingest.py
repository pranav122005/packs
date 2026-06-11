from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .utils import chunk_text


@dataclass
class DocumentChunk:
    text: str
    source_pdf: str
    page_number: int
    chunk_index: int


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        pages.append(extracted)
    return pages


def build_documents_from_pdf(pdf_path: Path, chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
    pages = extract_pdf_pages(pdf_path)
    docs: list[DocumentChunk] = []

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            docs.append(
                DocumentChunk(
                    text=chunk,
                    source_pdf=pdf_path.name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )
            )
    return docs


def iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    for path in sorted(pdf_dir.glob("*.pdf")):
        if path.is_file():
            yield path


def load_all_documents(pdf_dir: Path, chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
    all_docs: list[DocumentChunk] = []
    for pdf_path in iter_pdfs(pdf_dir):
        all_docs.extend(build_documents_from_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap))
    return all_docs
