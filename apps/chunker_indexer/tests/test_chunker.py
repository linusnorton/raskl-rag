"""Unit tests for heading-based semantic chunking."""

from __future__ import annotations

import pytest

from ras_chunker.chunker import chunk_blocks, _estimate_tokens
from ras_chunker.config import ChunkerConfig
from ras_chunker.loader import DocprocOutput, _FootnoteRecord, _TextBlock
from ras_chunker.schema import DocMeta, StitchedBlock


def _sb(
    text: str,
    block_type: str = "paragraph",
    page: int = 1,
    footnote_refs: list[int] | None = None,
    block_id: str | None = None,
) -> StitchedBlock:
    bid = block_id or f"blk-{hash(text) & 0xFFFFFF:06x}"
    return StitchedBlock(
        block_ids=[bid],
        doc_id="test-doc",
        start_page=page,
        end_page=page,
        text=text,
        block_type=block_type,
        footnote_refs=footnote_refs or [],
    )


def _make_output(
    footnotes: list[_FootnoteRecord] | None = None,
) -> DocprocOutput:
    out = object.__new__(DocprocOutput)
    out.doc_dir = None
    out.meta = DocMeta(doc_id="test-doc", source_filename="test.pdf", sha256_pdf="abc123")
    out.blocks = []
    out.footnotes = footnotes or []
    out.footnote_refs = []
    return out


class TestChunkBoundaries:
    def test_heading_starts_new_chunk(self):
        blocks = [
            _sb("Introduction", block_type="heading"),
            _sb("First paragraph of intro."),
            _sb("Methods", block_type="heading"),
            _sb("First paragraph of methods."),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert len(chunks) == 2
        assert chunks[0].section_heading == "Introduction"
        assert chunks[1].section_heading == "Methods"

    def test_max_tokens_splits_chunk(self):
        # Each block is ~50 tokens (200 chars / 4)
        long_text = "x" * 200
        blocks = [
            _sb("Heading", block_type="heading"),
            _sb(long_text),
            _sb(long_text),
            _sb(long_text),
        ]
        config = ChunkerConfig(max_chunk_tokens=80, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        # Should split after exceeding 80 tokens
        assert len(chunks) >= 2

    def test_small_chunks_merged(self):
        blocks = [
            _sb("Heading", block_type="heading"),
            _sb("Big paragraph. " * 40),  # ~150 tokens
            _sb("Tiny.", block_type="paragraph"),  # ~1 token
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=64)
        chunks = chunk_blocks(blocks, _make_output(), config)
        # The tiny chunk should be merged into the previous
        assert len(chunks) == 1

    def test_boilerplate_skipped(self):
        blocks = [
            _sb("Real content."),
            _sb("HEADER TEXT", block_type="header"),
            _sb("Page 42", block_type="page_number"),
            _sb("More content."),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert len(chunks) == 1
        assert "HEADER TEXT" not in chunks[0].text
        assert "Page 42" not in chunks[0].text

    def test_footnote_blocks_skipped(self):
        blocks = [
            _sb("Body text."),
            _sb("1 This is a footnote.", block_type="footnote"),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert len(chunks) == 1
        assert "This is a footnote" not in chunks[0].text


class TestFootnoteInlining:
    def test_footnotes_appended_to_chunk(self):
        blocks = [
            _sb("Text with reference.", footnote_refs=[1, 2]),
        ]
        footnotes = [
            _FootnoteRecord(footnote_id="fn1", doc_id="test-doc", page_num_1=1, footnote_number=1, text_raw="First note.", text_clean="First note."),
            _FootnoteRecord(footnote_id="fn2", doc_id="test-doc", page_num_1=1, footnote_number=2, text_raw="Second note.", text_clean="Second note."),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(footnotes), config)
        assert len(chunks) == 1
        assert "---" in chunks[0].text
        assert "Footnotes:" in chunks[0].text
        assert "[1] First note." in chunks[0].text
        assert "[2] Second note." in chunks[0].text

    def test_no_footnotes_when_no_refs(self):
        blocks = [_sb("Plain text.")]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert "Footnotes:" not in chunks[0].text


class TestChunkMetadata:
    def test_chunk_id_deterministic(self):
        blocks = [_sb("Hello world.")]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        c1 = chunk_blocks(blocks, _make_output(), config)
        c2 = chunk_blocks(blocks, _make_output(), config)
        assert c1[0].chunk_id == c2[0].chunk_id

    def test_page_range_from_blocks(self):
        blocks = [
            _sb("Start.", page=3),
            _sb("Middle.", page=4),
            _sb("End.", page=5),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert chunks[0].start_page == 3
        assert chunks[0].end_page == 5

    def test_chunk_index_sequential(self):
        blocks = [
            _sb("Heading A", block_type="heading"),
            _sb("Content A."),
            _sb("Heading B", block_type="heading"),
            _sb("Content B."),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert [c.chunk_index for c in chunks] == [0, 1]

    def test_empty_input(self):
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks([], _make_output(), config)
        assert chunks == []


class TestTokenEstimation:
    def test_basic_estimation(self):
        assert _estimate_tokens("hello") == 1  # 5 chars / 4 = 1
        assert _estimate_tokens("hello world test") == 4  # 16 chars / 4 = 4

    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # min 1
