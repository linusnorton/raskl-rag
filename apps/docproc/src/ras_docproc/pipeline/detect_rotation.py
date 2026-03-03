"""Pipeline step: detect vertical text and suggest page rotation."""

from __future__ import annotations

import logging

from ras_docproc.pipeline.extract_mupdf import MuPDFPageData
from ras_docproc.schema import PageRecord

logger = logging.getLogger(__name__)


def detect_rotation(
    mupdf_data: dict[int, MuPDFPageData],
    page_records: list[PageRecord],
) -> tuple[list[PageRecord], dict[int, int]]:
    """Detect vertical text and suggest rotation for each page.

    Computes vertical_text_ratio from MuPDF line direction vectors.
    dir ≈ (0, ±1) indicates vertical text.
    If vertical_text_ratio > 0.6, suggests rotation.

    Returns:
        (updated page_records, dict of page_num -> suggested_rotation_cw)
    """
    rotations: dict[int, int] = {}

    # Index page records by page number
    pr_index = {pr.page_num_1: pr for pr in page_records}

    for page_num, page_data in mupdf_data.items():
        pr = pr_index.get(page_num)
        if not pr:
            continue

        total = page_data.total_line_count
        vertical = page_data.vertical_line_count

        if total > 0:
            ratio = vertical / total
        else:
            ratio = 0.0

        pr.vertical_text_ratio = round(ratio, 4)

        if ratio > 0.6:
            pr.has_vertical_text = True
            # Determine rotation direction from span directions
            up_count = 0
            down_count = 0
            for span in page_data.spans:
                dx, dy = span.direction
                if abs(dx) < 0.3 and abs(dy) > 0.7:
                    if dy > 0:
                        down_count += 1
                    else:
                        up_count += 1

            # dir (0, 1) = text going down = needs 90° CW rotation
            # dir (0, -1) = text going up = needs 270° CW rotation
            if down_count >= up_count:
                pr.suggested_rotation_cw = 90
            else:
                pr.suggested_rotation_cw = 270

            rotations[page_num] = pr.suggested_rotation_cw
            logger.info(
                "Page %d: vertical_text_ratio=%.2f, suggested_rotation=%d°",
                page_num, ratio, pr.suggested_rotation_cw,
            )

    return page_records, rotations
