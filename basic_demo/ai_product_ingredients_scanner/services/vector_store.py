from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_store(index_path: Path, meta_path: Path, vectors: np.ndarray, chunks: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = vectors.astype("float32")
    if vectors.ndim != 2:
        raise ValueError("Vectors must be a 2D array.")

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(_normalize(vectors))
    faiss.write_index(faiss_index, str(index_path))

    meta_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def load_store(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def search(index, meta: list[dict[str, Any]], query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    if index is None or not meta:
        return []

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    query_vector = query_vector.astype("float32")
    query_vector = _normalize(query_vector)

    scores, ids = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    return results
