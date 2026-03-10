"""JSONL version diffing for versioned S3 output."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BlockDiff:
    """A single block that changed between versions."""

    block_id: str
    old_text: str
    new_text: str
    unified_diff: str


@dataclass
class DiffReport:
    """Summary of changes between two JSONL versions."""

    doc_id: str
    old_version: int
    new_version: int
    blocks_added: int = 0
    blocks_removed: int = 0
    blocks_changed: int = 0
    blocks_unchanged: int = 0
    changed_blocks: list[BlockDiff] = field(default_factory=list)
    meta_diff: str | None = None

    @property
    def has_changes(self) -> bool:
        return self.blocks_added > 0 or self.blocks_removed > 0 or self.blocks_changed > 0


def _load_blocks_from_jsonl(path: Path) -> dict[str, str]:
    """Load text_blocks.jsonl into {block_id: text_raw}."""
    blocks: dict[str, str] = {}
    if not path.exists():
        return blocks
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                blocks[record["block_id"]] = record.get("text_raw", "")
    return blocks


def _load_blocks_from_text(text: str) -> dict[str, str]:
    """Load text_blocks from a JSONL string into {block_id: text_raw}."""
    blocks: dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            blocks[record["block_id"]] = record.get("text_raw", "")
    return blocks


def _load_document_meta(path: Path) -> dict:
    """Load first line of documents.jsonl as dict."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        return json.loads(first_line) if first_line else {}


def _load_meta_from_text(text: str) -> dict:
    """Load first line of documents.jsonl from a string."""
    first_line = text.strip().split("\n", 1)[0].strip()
    return json.loads(first_line) if first_line else {}


def diff_versions(old_dir: Path, new_dir: Path, doc_id: str, old_version: int, new_version: int) -> DiffReport:
    """Compare two JSONL output directories and produce a DiffReport."""
    old_blocks = _load_blocks_from_jsonl(old_dir / "text_blocks.jsonl")
    new_blocks = _load_blocks_from_jsonl(new_dir / "text_blocks.jsonl")

    return _diff_blocks(
        old_blocks,
        new_blocks,
        doc_id,
        old_version,
        new_version,
        old_meta_text=None,
        new_meta_text=None,
        old_dir=old_dir,
        new_dir=new_dir,
    )


def diff_versions_from_text(
    old_blocks_text: str,
    new_blocks_text: str,
    old_meta_text: str | None,
    new_meta_text: str | None,
    doc_id: str,
    old_version: int,
    new_version: int,
) -> DiffReport:
    """Compare two JSONL strings and produce a DiffReport."""
    old_blocks = _load_blocks_from_text(old_blocks_text) if old_blocks_text else {}
    new_blocks = _load_blocks_from_text(new_blocks_text) if new_blocks_text else {}

    return _diff_blocks(
        old_blocks,
        new_blocks,
        doc_id,
        old_version,
        new_version,
        old_meta_text=old_meta_text,
        new_meta_text=new_meta_text,
    )


def _diff_blocks(
    old_blocks: dict[str, str],
    new_blocks: dict[str, str],
    doc_id: str,
    old_version: int,
    new_version: int,
    old_meta_text: str | None = None,
    new_meta_text: str | None = None,
    old_dir: Path | None = None,
    new_dir: Path | None = None,
) -> DiffReport:
    old_ids = set(old_blocks.keys())
    new_ids = set(new_blocks.keys())

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    changed_blocks: list[BlockDiff] = []
    unchanged = 0

    for block_id in sorted(common):
        old_text = old_blocks[block_id]
        new_text = new_blocks[block_id]
        if old_text != new_text:
            diff = "\n".join(
                difflib.unified_diff(
                    old_text.splitlines(),
                    new_text.splitlines(),
                    fromfile=f"v{old_version}",
                    tofile=f"v{new_version}",
                    lineterm="",
                )
            )
            changed_blocks.append(BlockDiff(block_id=block_id, old_text=old_text, new_text=new_text, unified_diff=diff))
        else:
            unchanged += 1

    # Diff document metadata
    meta_diff = None
    if old_meta_text is not None and new_meta_text is not None:
        old_meta = _load_meta_from_text(old_meta_text)
        new_meta = _load_meta_from_text(new_meta_text)
        for d in (old_meta, new_meta):
            d.pop("created_at", None)
            d.pop("source_path", None)
        if old_meta != new_meta:
            meta_diff = "\n".join(
                difflib.unified_diff(
                    json.dumps(old_meta, indent=2, default=str).splitlines(),
                    json.dumps(new_meta, indent=2, default=str).splitlines(),
                    fromfile=f"v{old_version}/documents.jsonl",
                    tofile=f"v{new_version}/documents.jsonl",
                    lineterm="",
                )
            )
    elif old_dir and new_dir:
        old_meta = _load_document_meta(old_dir / "documents.jsonl")
        new_meta = _load_document_meta(new_dir / "documents.jsonl")
        for d in (old_meta, new_meta):
            d.pop("created_at", None)
            d.pop("source_path", None)
        if old_meta != new_meta:
            meta_diff = "\n".join(
                difflib.unified_diff(
                    json.dumps(old_meta, indent=2, default=str).splitlines(),
                    json.dumps(new_meta, indent=2, default=str).splitlines(),
                    fromfile=f"v{old_version}/documents.jsonl",
                    tofile=f"v{new_version}/documents.jsonl",
                    lineterm="",
                )
            )

    return DiffReport(
        doc_id=doc_id,
        old_version=old_version,
        new_version=new_version,
        blocks_added=len(added),
        blocks_removed=len(removed),
        blocks_changed=len(changed_blocks),
        blocks_unchanged=unchanged,
        changed_blocks=changed_blocks,
        meta_diff=meta_diff,
    )
