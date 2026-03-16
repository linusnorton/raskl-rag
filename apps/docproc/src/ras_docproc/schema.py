"""Pydantic v2 data models for document processing records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box in PDF coordinate space (top-left origin)."""

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

    @property
    def area(self) -> float:
        return self.width * self.height


class MetadataSource(BaseModel):
    """Provenance tracking for a metadata field."""

    field: str
    source: str  # e.g. "pdf_metadata", "cover_page_regex", "llm_extraction", "crossref", "openlibrary", "web_search"
    confidence: float = 1.0
    raw_value: str = ""


class DocumentRecord(BaseModel):
    """Document-level metadata."""

    doc_id: str
    source_filename: str
    sha256_pdf: str
    created_at: datetime = Field(default_factory=datetime.now)
    title: str | None = None
    author: str | None = None
    editor: str | None = None
    publication: str | None = None
    journal_ref: str | None = None
    volume: str | None = None
    issue: str | None = None
    page_range_label: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    document_type: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    language: str | None = None
    isbn: str | None = None
    issn: str | None = None
    series: str | None = None
    description: str | None = None
    metadata_sources: list[MetadataSource] = Field(default_factory=list)
    page_offset: int = 0
    extraction_version: str = "0.1.0"
    config_hash: str | None = None


BlockType = Literal[
    "paragraph",
    "heading",
    "list_item",
    "table",
    "caption",
    "footnote",
    "header",
    "footer",
    "page_number",
    "unknown",
]


class PageRecord(BaseModel):
    """Per-page metadata."""

    doc_id: str
    page_index_0: int
    page_num_1: int
    width: float
    height: float
    page_rotation: int = 0
    page_hash: str | None = None
    text_char_count: int = 0
    image_count: int = 0
    content_bbox: BBox | None = None
    has_vertical_text: bool = False
    vertical_text_ratio: float = 0.0
    suggested_rotation_cw: int = 0


class TextBlockRecord(BaseModel):
    """A text block extracted from a page."""

    block_id: str
    doc_id: str
    page_num_1: int
    bbox: BBox
    text_raw: str
    text_clean: str = ""
    block_type: BlockType = "paragraph"
    section_path: str | None = None
    lang: str | None = None
    lang_confidence: float | None = None
    lang_candidates: list[str] = Field(default_factory=list)
    reading_order: int = 0
    links: list[str] = Field(default_factory=list)


class FootnoteRecord(BaseModel):
    """A detected footnote."""

    footnote_id: str
    doc_id: str
    page_num_1: int
    footnote_number: int
    bbox: BBox
    text_raw: str
    text_clean: str = ""
    footnote_type: Literal["citation", "explanatory", "mixed"] = "explanatory"
    lang: str | None = None
    lang_confidence: float | None = None


class FootnoteRefRecord(BaseModel):
    """A link between a body text block and a footnote."""

    ref_id: str
    doc_id: str
    page_num_1: int
    parent_block_id: str
    footnote_number: int
    footnote_id: str | None = None
    match_type: str = ""
    evidence_span_bbox: BBox | None = None
    context_snippet: str = ""


class FigureRecord(BaseModel):
    """An extracted figure/image."""

    figure_id: str
    doc_id: str
    page_num_1: int
    figure_index_on_page: int = 0
    bbox: BBox | None = None
    asset_original_path: str | None = None
    asset_jpg_path: str | None = None
    asset_thumb_path: str | None = None
    asset_sha256: str | None = None
    caption_block_ids: list[str] = Field(default_factory=list)
    caption_text_raw: str = ""
    caption_text_clean: str = ""
    lang: str | None = None
    lang_confidence: float | None = None
    plate_id: str | None = None
    derived_from: str | None = None
    applied_rotation_cw: int = 0


class PlateRecord(BaseModel):
    """A plate page containing multiple figures."""

    plate_id: str
    doc_id: str
    page_num_1: int
    plate_label: str = ""
    figure_ids: list[str] = Field(default_factory=list)
    plate_note_block_ids: list[str] = Field(default_factory=list)
    plate_note_text_raw: str = ""
    plate_note_text_clean: str = ""
