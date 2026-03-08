"""Unit tests for cross-page paragraph restitching."""

from __future__ import annotations

import pytest

from ras_chunker.loader import DocprocOutput, _FootnoteRefRecord, _TextBlock
from ras_chunker.restitch import restitch


def _make_block(
    page: int,
    order: int,
    text: str,
    block_type: str = "paragraph",
    block_id: str | None = None,
) -> _TextBlock:
    bid = block_id or f"blk-p{page}-{order}"
    return _TextBlock(
        block_id=bid,
        doc_id="test-doc",
        page_num_1=page,
        text_raw=text,
        text_clean=text,
        block_type=block_type,
        reading_order=order,
    )


def _make_output(blocks: list[_TextBlock], refs: list[_FootnoteRefRecord] | None = None) -> DocprocOutput:
    """Create a minimal DocprocOutput without touching the filesystem."""
    out = object.__new__(DocprocOutput)
    out.doc_dir = None
    out.meta = None
    out.blocks = blocks
    out.footnotes = []
    out.footnote_refs = refs or []
    return out


class TestRestitchMerging:
    def test_merge_continuation_across_pages(self):
        """Paragraph ending without punctuation + next page starting lowercase → merge."""
        blocks = [
            _make_block(1, 0, "The expedition continued through the"),
            _make_block(2, 0, "dense jungle until they reached the river."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 1
        assert result[0].text == "The expedition continued through the dense jungle until they reached the river."
        assert result[0].start_page == 1
        assert result[0].end_page == 2
        assert len(result[0].block_ids) == 2

    def test_no_merge_when_sentence_ends(self):
        """Paragraph ending with period → no merge."""
        blocks = [
            _make_block(1, 0, "The expedition ended here."),
            _make_block(2, 0, "next day they departed."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 2

    def test_no_merge_when_next_starts_uppercase(self):
        """Next paragraph starting with uppercase → no merge (new sentence)."""
        blocks = [
            _make_block(1, 0, "The expedition continued through the"),
            _make_block(2, 0, "The next morning they departed."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 2

    def test_no_merge_across_non_consecutive_pages(self):
        """Pages 1 and 3 (skipping 2) → no merge."""
        blocks = [
            _make_block(1, 0, "The expedition continued through the"),
            _make_block(3, 0, "dense jungle until they reached the river."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 2

    def test_no_merge_heading_into_paragraph(self):
        """Heading should not be merged with a paragraph."""
        blocks = [
            _make_block(1, 0, "Chapter Title", block_type="heading"),
            _make_block(2, 0, "the story begins here."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 2

    def test_chain_merge_across_three_pages(self):
        """Chain merge: page 1 → 2 → 3."""
        blocks = [
            _make_block(1, 0, "The long paragraph begins here and"),
            _make_block(2, 0, "continues on the second page and"),
            _make_block(3, 0, "finally ends on the third page."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 1
        assert "begins here" in result[0].text
        assert "second page" in result[0].text
        assert "third page" in result[0].text
        assert result[0].start_page == 1
        assert result[0].end_page == 3

    def test_merge_preserves_non_merged_blocks(self):
        """Blocks that don't merge should pass through unchanged."""
        blocks = [
            _make_block(1, 0, "Chapter One", block_type="heading"),
            _make_block(1, 1, "First paragraph."),
            _make_block(2, 0, "Second paragraph."),
        ]
        result = restitch(_make_output(blocks))
        assert len(result) == 3
        types = [r.block_type for r in result]
        assert types == ["heading", "paragraph", "paragraph"]

    def test_no_merge_when_not_first_on_page(self):
        """Only merge if the continuation is the first block on the next page."""
        blocks = [
            _make_block(1, 0, "The expedition continued through the"),
            _make_block(2, 0, "A heading here.", block_type="heading"),
            _make_block(2, 1, "lowercase continuation."),
        ]
        result = restitch(_make_output(blocks))
        # The first block shouldn't merge with block at order=1 on page 2
        assert len(result) == 3

    def test_empty_input(self):
        result = restitch(_make_output([]))
        assert result == []

    def test_single_block(self):
        blocks = [_make_block(1, 0, "Single paragraph.")]
        result = restitch(_make_output(blocks))
        assert len(result) == 1


class TestRestitchFootnoteRefs:
    def test_footnote_refs_collected(self):
        blocks = [_make_block(1, 0, "Text with ref.", block_id="blk-1")]
        refs = [
            _FootnoteRefRecord(
                ref_id="ref-1",
                doc_id="test-doc",
                page_num_1=1,
                parent_block_id="blk-1",
                footnote_number=1,
                footnote_id="fn-abc",
            )
        ]
        result = restitch(_make_output(blocks, refs))
        assert result[0].footnote_refs == ["fn-abc"]

    def test_merged_blocks_combine_refs(self):
        blocks = [
            _make_block(1, 0, "First part with ref", block_id="blk-1"),
            _make_block(2, 0, "second part with ref.", block_id="blk-2"),
        ]
        refs = [
            _FootnoteRefRecord(ref_id="r1", doc_id="test-doc", page_num_1=1, parent_block_id="blk-1", footnote_number=1, footnote_id="fn-aaa"),
            _FootnoteRefRecord(ref_id="r2", doc_id="test-doc", page_num_1=2, parent_block_id="blk-2", footnote_number=3, footnote_id="fn-bbb"),
        ]
        result = restitch(_make_output(blocks, refs))
        assert len(result) == 1
        assert result[0].footnote_refs == ["fn-aaa", "fn-bbb"]
