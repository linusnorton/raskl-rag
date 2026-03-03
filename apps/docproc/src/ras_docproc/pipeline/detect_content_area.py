"""Pipeline step: compute per-page content area bounding box."""

from __future__ import annotations

import logging

from ras_docproc.schema import BBox, PageRecord, TextBlockRecord

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile value from a sorted list."""
    if not values:
        return 0.0
    values = sorted(values)
    idx = pct / 100.0 * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def detect_content_area(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    page_records: list[PageRecord],
) -> list[PageRecord]:
    """Compute content_bbox for each page from remaining body block bounding boxes.

    Uses 5th/95th percentile of block coordinates for robustness against outliers.
    Should be called after boilerplate removal so only body content remains.
    """
    # Index page records by page number for fast lookup
    page_index: dict[int, PageRecord] = {pr.page_num_1: pr for pr in page_records}

    computed = 0
    for page_num, blocks in blocks_by_page.items():
        # Only use body blocks (skip header/footer/page_number)
        body_blocks = [b for b in blocks if b.block_type not in ("header", "footer", "page_number")]
        if len(body_blocks) < 2:
            continue

        x0_vals = [b.bbox.x0 for b in body_blocks]
        y0_vals = [b.bbox.y0 for b in body_blocks]
        x1_vals = [b.bbox.x1 for b in body_blocks]
        y1_vals = [b.bbox.y1 for b in body_blocks]

        content_bbox = BBox(
            x0=_percentile(x0_vals, 5),
            y0=_percentile(y0_vals, 5),
            x1=_percentile(x1_vals, 95),
            y1=_percentile(y1_vals, 95),
        )

        pr = page_index.get(page_num)
        if pr:
            pr.content_bbox = content_bbox
            computed += 1

    logger.info("Computed content_bbox for %d/%d pages", computed, len(page_records))
    return page_records
