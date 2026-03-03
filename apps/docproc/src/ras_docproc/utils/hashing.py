"""Hashing and ID generation utilities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def file_sha256(path: Path | str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(filename: str) -> str:
    """Create a URL-safe slug from a filename (without extension)."""
    name = Path(filename).stem
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    return name


def make_doc_id(filename: str, sha256: str) -> str:
    """Generate a deterministic document ID from filename slug and SHA256 prefix."""
    return f"{slug(filename)}-{sha256[:12]}"


def text_hash(text: str) -> str:
    """SHA256 hex digest of normalized text."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_block_id(doc_id: str, page_num: int, bbox_str: str, kind: str, txt_hash: str) -> str:
    """Generate a deterministic block ID."""
    raw = f"{doc_id}:{page_num}:{bbox_str}:{kind}:{txt_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def page_content_hash(block_hashes: list[str]) -> str:
    """Hash of sorted block hashes to detect page content changes."""
    combined = "|".join(sorted(block_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
