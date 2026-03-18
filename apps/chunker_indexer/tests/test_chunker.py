"""Unit tests for heading-based semantic chunking."""

from __future__ import annotations

import pytest

from ras_chunker.chunker import chunk_blocks, _estimate_tokens
from ras_chunker.config import ChunkerConfig
from ras_chunker.loader import DocprocOutput, _FootnoteRecord, _TextBlock
from ras_chunker.page_filter import filter_blocks_by_article_range, parse_page_range, filter_footnotes_by_page_range
from ras_chunker.schema import DocMeta, StitchedBlock


def _sb(
    text: str,
    block_type: str = "paragraph",
    page: int = 1,
    footnote_refs: list[str] | None = None,
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
    page_range_label: str | None = None,
    title: str | None = None,
) -> DocprocOutput:
    out = object.__new__(DocprocOutput)
    out.doc_dir = None
    out.meta = DocMeta(
        doc_id="test-doc",
        source_filename="test.pdf",
        sha256_pdf="abc123",
        page_range_label=page_range_label,
        title=title,
    )
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
            _sb(long_text, block_id="b1"),
            _sb(long_text, block_id="b2"),
            _sb(long_text, block_id="b3"),
        ]
        config = ChunkerConfig(max_chunk_tokens=80, min_chunk_tokens=1, overlap_tokens=50)
        chunks = chunk_blocks(blocks, _make_output(), config)
        # Should split after exceeding 80 tokens
        assert len(chunks) >= 2
        # Second chunk should contain overlap text from the first chunk
        assert long_text in chunks[1].text
        # Overlap block_ids should NOT appear in the second chunk's block_ids
        second_block_ids = chunks[1].block_ids
        assert "b1" not in second_block_ids

    def test_no_overlap_on_heading_split(self):
        """Heading boundaries should NOT carry overlap from the previous chunk."""
        blocks = [
            _sb("Heading A", block_type="heading"),
            _sb("Content A paragraph one."),
            _sb("Content A paragraph two."),
            _sb("Heading B", block_type="heading"),
            _sb("Content B paragraph one."),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1, overlap_tokens=128)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert len(chunks) == 2
        # Second chunk should NOT contain text from first chunk
        assert "Content A" not in chunks[1].text

    def test_overlap_disabled_when_zero(self):
        """overlap_tokens=0 should produce no overlap (backward compat)."""
        long_text = "x" * 200
        blocks = [
            _sb(long_text, block_id="b1"),
            _sb(long_text, block_id="b2"),
            _sb(long_text, block_id="b3"),
        ]
        config = ChunkerConfig(max_chunk_tokens=80, min_chunk_tokens=1, overlap_tokens=0)
        chunks = chunk_blocks(blocks, _make_output(), config)
        assert len(chunks) >= 2
        # No overlap: second chunk should not contain first chunk's block_id
        assert "b1" not in chunks[1].block_ids

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
            _sb("Text with reference.", footnote_refs=["fn1", "fn2"]),
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

    def test_citation_footnote_gets_cites_marker(self):
        blocks = [
            _sb("Text with reference.", footnote_refs=["fn1", "fn2", "fn3"]),
        ]
        footnotes = [
            _FootnoteRecord(footnote_id="fn1", doc_id="test-doc", page_num_1=1, footnote_number=1, text_raw="Gullick (1992: 246).", text_clean="Gullick (1992: 246).", footnote_type="citation"),
            _FootnoteRecord(footnote_id="fn2", doc_id="test-doc", page_num_1=1, footnote_number=2, text_raw="Head of sub-district.", text_clean="Head of sub-district.", footnote_type="explanatory"),
            _FootnoteRecord(footnote_id="fn3", doc_id="test-doc", page_num_1=1, footnote_number=3, text_raw="Ibid. This reflects class divisions.", text_clean="Ibid. This reflects class divisions.", footnote_type="mixed"),
        ]
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, _make_output(footnotes), config)
        assert "[1] [cites:] Gullick (1992: 246)." in chunks[0].text
        assert "[2] Head of sub-district." in chunks[0].text
        assert "[3] [cites:] Ibid. This reflects class divisions." in chunks[0].text


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


class TestPageRangeParsing:
    def test_parse_standard_range(self):
        assert parse_page_range("57-61") == (57, 61)

    def test_parse_en_dash(self):
        assert parse_page_range("57–61") == (57, 61)

    def test_parse_single_page(self):
        assert parse_page_range("42") == (42, 42)

    def test_parse_none(self):
        assert parse_page_range(None) is None

    def test_parse_empty(self):
        assert parse_page_range("") is None

    def test_parse_with_spaces(self):
        assert parse_page_range(" 57 - 61 ") == (57, 61)


class TestArticleRangeFiltering:
    def test_blocks_outside_page_range_filtered(self):
        blocks = [
            _sb("Cover page content", page=1),  # JSTOR cover — outside range
            _sb("Previous article ending", page=56),  # before range
            _sb("Article start", page=57),
            _sb("Article middle", page=59),
            _sb("Article end", page=61),
            _sb("Next article start", page=62),  # after range
        ]
        result = filter_blocks_by_article_range(blocks, "57-61", None)
        assert len(result) == 3
        assert all(57 <= b.start_page <= 61 for b in result)

    def test_first_page_trimmed_at_title(self):
        blocks = [
            _sb("End of previous article.", page=57),
            _sb("The Red and White Flag Societies", block_type="heading", page=57),
            _sb("This article discusses...", page=57),
            _sb("More content.", page=58),
        ]
        result = filter_blocks_by_article_range(
            blocks, "57-61", "The Red and White Flag Societies"
        )
        assert len(result) == 3
        assert result[0].text == "The Red and White Flag Societies"

    def test_no_filtering_without_page_range(self):
        blocks = [
            _sb("Page 1 content", page=1),
            _sb("Page 2 content", page=2),
        ]
        result = filter_blocks_by_article_range(blocks, None, "Some Title")
        assert len(result) == 2

    def test_no_trimming_when_title_not_found(self):
        blocks = [
            _sb("Previous article ending", page=57),
            _sb("Unrelated heading", block_type="heading", page=57),
            _sb("Content on page 57", page=57),
            _sb("Content on page 58", page=58),
        ]
        result = filter_blocks_by_article_range(
            blocks, "57-61", "Completely Different Title"
        )
        # All blocks within range kept — no title-based trimming
        assert len(result) == 4

    def test_title_fuzzy_match(self):
        """OCR variations in title should still match."""
        blocks = [
            _sb("Previous article text.", page=57),
            _sb("The Red & White Flag Societies", block_type="heading", page=57),
            _sb("Article body.", page=57),
        ]
        result = filter_blocks_by_article_range(
            blocks, "57-61", "The Red and White Flag Societies"
        )
        assert len(result) == 2
        assert result[0].block_type == "heading"

    def test_footnotes_outside_range_filtered(self):
        footnotes = [
            _FootnoteRecord(footnote_id="fn1", doc_id="test-doc", page_num_1=56, footnote_number=1, text_raw="Bleed footnote."),
            _FootnoteRecord(footnote_id="fn2", doc_id="test-doc", page_num_1=57, footnote_number=1, text_raw="Article footnote."),
            _FootnoteRecord(footnote_id="fn3", doc_id="test-doc", page_num_1=62, footnote_number=1, text_raw="Next article footnote."),
        ]
        result = filter_footnotes_by_page_range(footnotes, "57-61")
        assert len(result) == 1
        assert result[0].footnote_id == "fn2"

    def test_footnotes_no_filtering_without_range(self):
        footnotes = [
            _FootnoteRecord(footnote_id="fn1", doc_id="test-doc", page_num_1=1, footnote_number=1, text_raw="Note."),
        ]
        result = filter_footnotes_by_page_range(footnotes, None)
        assert len(result) == 1

    def test_chunk_blocks_applies_page_filter(self):
        """Integration: chunk_blocks respects page_range_label from DocMeta."""
        blocks = [
            _sb("Cover page.", page=1),
            _sb("Article content.", page=57),
        ]
        output = _make_output(page_range_label="57-61")
        config = ChunkerConfig(max_chunk_tokens=1000, min_chunk_tokens=1)
        chunks = chunk_blocks(blocks, output, config)
        assert len(chunks) == 1
        assert "Cover page" not in chunks[0].text
        assert "Article content" in chunks[0].text
