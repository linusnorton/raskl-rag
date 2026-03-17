"""Load docproc JSONL output into memory."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from .schema import DocMeta


# Lightweight record types — we only need the fields the chunker uses,
# and we avoid importing from ras_docproc so the package stays standalone.

class _TextBlock(BaseModel):
    block_id: str
    doc_id: str
    page_num_1: int
    text_raw: str
    text_clean: str = ""
    block_type: str = "paragraph"
    section_path: str | None = None
    lang: str | None = None
    reading_order: int = 0
    links: list[str] = []


class _FootnoteRecord(BaseModel):
    footnote_id: str
    doc_id: str
    page_num_1: int
    footnote_number: int
    text_raw: str
    text_clean: str = ""
    footnote_type: str = "explanatory"


class _FootnoteRefRecord(BaseModel):
    ref_id: str
    doc_id: str
    page_num_1: int
    parent_block_id: str
    footnote_number: int
    footnote_id: str | None = None


class _BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


# Minimum dimension (PDF points) for a figure to be indexed.
# Filters out logos, icons, and journal cover thumbnails.
_MIN_FIGURE_DIMENSION = 150  # ~2 inches


class _FigureRecord(BaseModel):
    figure_id: str
    doc_id: str
    page_num_1: int
    bbox: _BBox | None = None
    asset_jpg_path: str | None = None
    asset_thumb_path: str | None = None
    caption_text_clean: str = ""
    derived_from: str | None = None


def _is_substantial_figure(fig: _FigureRecord) -> bool:
    """Filter out rendered clips, logos, icons, and journal cover thumbnails."""
    if fig.derived_from == "rendered_clip":
        return False
    # Page 1 uncaptioned images are JSTOR/Project MUSE cover elements (logo, journal cover)
    if fig.page_num_1 == 1 and not fig.caption_text_clean:
        return False
    if fig.bbox is None:
        return True  # No bbox info — keep by default
    return fig.bbox.width >= _MIN_FIGURE_DIMENSION and fig.bbox.height >= _MIN_FIGURE_DIMENSION


def _read_jsonl(path: Path, model_class: type) -> list:
    adapter = TypeAdapter(model_class)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(adapter.validate_json(line))
    return records


class DocprocOutput:
    """All docproc output for a single document."""

    def __init__(self, doc_dir: Path) -> None:
        self.doc_dir = doc_dir

        # Load document metadata
        docs_path = doc_dir / "documents.jsonl"
        raw = json.loads(docs_path.read_text().strip().splitlines()[0])
        self.meta = DocMeta(
            doc_id=raw["doc_id"],
            source_filename=raw["source_filename"],
            title=raw.get("title"),
            author=raw.get("author"),
            year=raw.get("year"),
            publication=raw.get("publication"),
            document_type=raw.get("document_type"),
            page_offset=raw.get("page_offset", 0),
            sha256_pdf=raw["sha256_pdf"],
        )

        # Load text blocks
        self.blocks: list[_TextBlock] = _read_jsonl(doc_dir / "text_blocks.jsonl", _TextBlock)

        # Load footnotes (optional — may not exist)
        fn_path = doc_dir / "footnotes.jsonl"
        self.footnotes: list[_FootnoteRecord] = _read_jsonl(fn_path, _FootnoteRecord) if fn_path.exists() else []

        # Load footnote refs (optional)
        ref_path = doc_dir / "footnote_refs.jsonl"
        self.footnote_refs: list[_FootnoteRefRecord] = (
            _read_jsonl(ref_path, _FootnoteRefRecord) if ref_path.exists() else []
        )

        # Load figures (optional) — filter out rendered clips and small images (logos, icons)
        fig_path = doc_dir / "figures.jsonl"
        self.figures: list[_FigureRecord] = []
        if fig_path.exists():
            all_figs = _read_jsonl(fig_path, _FigureRecord)
            self.figures = [f for f in all_figs if _is_substantial_figure(f)]

    @property
    def doc_id(self) -> str:
        return self.meta.doc_id


def find_doc_dir(data_dir: Path, doc_id: str) -> Path:
    """Resolve a doc_id to its output directory under data_dir/out/."""
    doc_dir = data_dir / "out" / doc_id
    if not doc_dir.is_dir():
        raise FileNotFoundError(f"No output directory found: {doc_dir}")
    return doc_dir
