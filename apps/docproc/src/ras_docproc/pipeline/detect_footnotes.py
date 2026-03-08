"""Pipeline step: detect footnote blocks in the page footer zone."""

from __future__ import annotations

import logging
import re

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, FootnoteRecord, TextBlockRecord
from ras_docproc.utils.geometry import is_in_zone
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Patterns for footnote number at start of text
FOOTNOTE_PATTERNS = [
    re.compile(r"^(\d{1,3})\s"),         # "1 Some footnote text"
    re.compile(r"^\((\d{1,3})\)\s*"),     # "(1) Some footnote text"
    re.compile(r"^(\d{1,3})\.\s"),        # "1. Some footnote text"
    re.compile(r"^(\d{1,3})\)[\s]"),      # "1) Some footnote text"
]


def _is_full_page_bbox(bbox: BBox, page_h: float, tol: float = 2.0) -> bool:
    """Check if bbox spans the full page height (degenerate Qwen3 VL bbox)."""
    return bbox.y0 < tol and bbox.y1 > page_h - tol


def detect_footnotes(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    page_heights: dict[int, float],
    config: PipelineConfig,
    doc_id: str,
) -> tuple[dict[int, list[TextBlockRecord]], list[FootnoteRecord]]:
    """Detect footnote blocks in the footnote zone of each page.

    Returns:
        (updated blocks_by_page with block_type set, list of FootnoteRecords)
    """
    all_footnotes: list[FootnoteRecord] = []

    for page_num, blocks in blocks_by_page.items():
        page_h = page_heights.get(page_num, 800.0)

        for block in blocks:
            if block.block_type == "footnote":
                # Already classified (e.g., by Docling)
                fn_num = _extract_footnote_number(block.text_clean or block.text_raw)
                if fn_num is not None:
                    fn_id = make_block_id(doc_id, page_num, "fn", "footnote", text_hash(block.text_raw))
                    all_footnotes.append(
                        FootnoteRecord(
                            footnote_id=fn_id,
                            doc_id=doc_id,
                            page_num_1=page_num,
                            footnote_number=fn_num,
                            bbox=block.bbox,
                            text_raw=block.text_raw,
                            text_clean=block.text_clean,
                        )
                    )
                continue

            # Check if block is in footnote zone
            in_zone = is_in_zone(block.bbox, page_h, config.footnote_zone_top, config.footnote_zone_bottom)

            # Fallback: when bbox spans the full page (degenerate Qwen3 VL bbox),
            # zone check is meaningless — classify purely by text pattern.
            text_fallback = (
                not in_zone
                and block.block_type == "paragraph"
                and _is_full_page_bbox(block.bbox, page_h)
            )

            if not in_zone and not text_fallback:
                continue

            text = block.text_clean or block.text_raw
            fn_num = _extract_footnote_number(text)
            if fn_num is not None:
                block.block_type = "footnote"
                fn_id = make_block_id(doc_id, page_num, "fn", "footnote", text_hash(block.text_raw))
                all_footnotes.append(
                    FootnoteRecord(
                        footnote_id=fn_id,
                        doc_id=doc_id,
                        page_num_1=page_num,
                        footnote_number=fn_num,
                        bbox=block.bbox,
                        text_raw=block.text_raw,
                        text_clean=block.text_clean,
                    )
                )

    logger.info("Detected %d footnotes across %d pages", len(all_footnotes), len(blocks_by_page))
    return blocks_by_page, all_footnotes


def _extract_footnote_number(text: str) -> int | None:
    """Try to extract a footnote number from the start of text."""
    text = text.strip()
    for pattern in FOOTNOTE_PATTERNS:
        m = pattern.match(text)
        if m:
            return int(m.group(1))
    return None
