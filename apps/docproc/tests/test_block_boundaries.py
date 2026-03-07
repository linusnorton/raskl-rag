"""Regression tests for OCR block boundaries in processed output.

These tests validate the already-processed JSONL output in data/out/,
ensuring the Qwen3 VL prompt produces correct paragraph boundaries.
They skip if the processed output doesn't exist (e.g. in CI without PDFs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ras_docproc.schema import TextBlockRecord
from ras_docproc.utils.io import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_OUT = REPO_ROOT / "data" / "out"
SWETTENHAM_DOC_ID = "swettenham-journal-1874-1876-bbfb9df1239d"
SWETTENHAM_BLOCKS = DATA_OUT / SWETTENHAM_DOC_ID / "text_blocks.jsonl"


@pytest.fixture
def swettenham_blocks() -> list[TextBlockRecord]:
    if not SWETTENHAM_BLOCKS.exists():
        pytest.skip(f"Processed output not found: {SWETTENHAM_BLOCKS}")
    return read_jsonl(SWETTENHAM_BLOCKS, TextBlockRecord)


class TestSwettenhamBlockBoundaries:
    """Verify date entries in the Swettenham Journal get their own blocks."""

    def test_6th_october_is_separate_block(self, swettenham_blocks: list[TextBlockRecord]):
        """Page 111: '6th October.' must be a separate block, not merged with prior text."""
        page_111_blocks = [b for b in swettenham_blocks if b.page_num_1 == 111]
        assert len(page_111_blocks) > 0, "No blocks found on page 111"

        october_blocks = [
            b for b in page_111_blocks
            if "6th October" in (b.text_clean or b.text_raw)
        ]
        assert len(october_blocks) >= 1, (
            f"No block containing '6th October' found on page 111. "
            f"Blocks: {[b.text_clean[:80] for b in page_111_blocks]}"
        )

        # The block should START with the date entry (possibly bold-formatted),
        # not have it buried mid-paragraph
        oct_block = october_blocks[0]
        text = oct_block.text_clean or oct_block.text_raw
        assert text.strip().startswith("6th October") or text.strip().startswith("**6th October"), (
            f"'6th October' block should start with the date entry, got: {text[:120]}"
        )

    def test_date_entries_not_merged_with_prior_text(self, swettenham_blocks: list[TextBlockRecord]):
        """Date entries like '6th October.' should not be appended to the end of a prior block."""
        page_111_blocks = [b for b in swettenham_blocks if b.page_num_1 == 111]

        for block in page_111_blocks:
            text = block.text_clean or block.text_raw
            # If a block contains '6th October', it should not also contain
            # unrelated preceding narrative (e.g. about leaving at 11 A.M.)
            if "6th October" in text and "11 A.M" in text:
                pytest.fail(
                    f"'6th October' is merged with prior narrative in same block: {text[:200]}"
                )

    def test_messy_pdf_has_text_blocks(self, swettenham_blocks: list[TextBlockRecord]):
        """Ensure the qwen3vl backend produced non-empty output for the messy PDF."""
        assert len(swettenham_blocks) > 100, (
            f"Expected >100 text blocks from 151-page Swettenham Journal, got {len(swettenham_blocks)}"
        )
