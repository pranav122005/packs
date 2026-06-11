from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_allowed_image(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_EXTENSIONS


def save_upload(file_storage, upload_dir: Path) -> Path:
    ensure_dir(upload_dir)
    filename = secure_filename(file_storage.filename or "upload.jpg")
    target = upload_dir / filename
    file_storage.save(target)
    return target


def save_base64_image(data_url: str, upload_dir: Path, prefix: str = "webcam") -> Path:
    ensure_dir(upload_dir)
    match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid image data.")
    mime = match.group(1).split("/")[-1].lower()
    ext = "jpg" if mime == "jpeg" else mime
    raw = base64.b64decode(match.group(2))
    target = upload_dir / f"{prefix}.{ext}"
    target.write_bytes(raw)
    return target


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def safe_json_loads(value: str, default: Any = None) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return default
