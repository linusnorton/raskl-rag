"""Data models for the chunker/indexer pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StitchedBlock(BaseModel):
    """A text block that may span multiple pages after restitching."""

    block_ids: list[str]
    doc_id: str
    start_page: int
    end_page: int
    text: str
    block_type: str = "paragraph"
    section_path: str | None = None
    lang: str | None = None
    reading_order: int = 0
    footnote_refs: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A semantic chunk ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    start_page: int
    end_page: int
    section_heading: str | None = None
    text: str
    block_ids: list[str]
    token_count: int


class DocMeta(BaseModel):
    """Document metadata loaded from docproc output."""

    doc_id: str
    source_filename: str
    title: str | None = None
    author: str | None = None
    year: int | None = None
    publication: str | None = None
    document_type: str | None = None
    page_offset: int = 0
    sha256_pdf: str
    s3_prefix: str = ""


class FigureMeta(BaseModel):
    """Figure metadata for indexing."""

    figure_id: str
    doc_id: str
    page_num: int
    caption: str = ""
    asset_path: str
    thumb_path: str = ""
