"""Pipeline step: detect and remove header/footer boilerplate."""

from __future__ import annotations

import logging
import re
from collections import Counter

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.geometry import is_in_zone
from ras_docproc.utils.text import normalize_for_frequency

logger = logging.getLogger(__name__)

# Known platform headings that appear on cover pages (not real document content)
PLATFORM_HEADINGS = {"PROJECT MUSE", "JSTOR"}


def detect_boilerplate(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    page_heights: dict[int, float],
    config: PipelineConfig,
) -> tuple[dict[int, list[TextBlockRecord]], list[TextBlockRecord]]:
    """Detect and remove header/footer boilerplate.

    Phase 1: classify blocks in header/footer zones by geometry.
    Phase 2: compute normalized-line frequencies, mark lines appearing
             on > threshold fraction of pages as boilerplate.

    Returns:
        (filtered_blocks_by_page, removed_blocks)
    """
    total_pages = len(blocks_by_page)
    if total_pages == 0:
        return blocks_by_page, []

    # Phase 1: Classify blocks by zone
    for page_num, blocks in blocks_by_page.items():
        page_h = page_heights.get(page_num, 800.0)
        for block in blocks:
            if block.block_type in ("header", "footer"):
                continue  # Already classified by Docling
            if is_in_zone(block.bbox, page_h, 0.0, config.header_zone_bottom):
                block.block_type = "header"
            elif is_in_zone(block.bbox, page_h, config.footer_zone_top, 1.0):
                block.block_type = "footer"

    # Phase 2: Frequency-based boilerplate detection
    line_page_counts: Counter[str] = Counter()
    for page_num, blocks in blocks_by_page.items():
        page_lines: set[str] = set()
        for block in blocks:
            for line in block.text_clean.splitlines():
                normalized = normalize_for_frequency(line)
                if normalized and len(normalized) > 3:
                    page_lines.add(normalized)
        for line in page_lines:
            line_page_counts[line] += 1

    threshold = max(2, int(total_pages * config.boilerplate_threshold))
    boilerplate_lines = {line for line, count in line_page_counts.items() if count >= threshold}

    if boilerplate_lines:
        logger.info("Found %d boilerplate line patterns (threshold=%d/%d pages)", len(boilerplate_lines), threshold, total_pages)

    # Mark and remove boilerplate blocks
    removed: list[TextBlockRecord] = []
    filtered: dict[int, list[TextBlockRecord]] = {}

    for page_num, blocks in blocks_by_page.items():
        kept: list[TextBlockRecord] = []
        for block in blocks:
            is_boilerplate = False

            # Check if block text matches boilerplate patterns
            for line in block.text_clean.splitlines():
                normalized = normalize_for_frequency(line)
                if normalized in boilerplate_lines:
                    is_boilerplate = True
                    break

            # Remove known platform headings (e.g. "PROJECT MUSE®" on cover pages)
            if not is_boilerplate:
                text_normalized = re.sub(r"[^\w\s]", "", block.text_clean.strip()).strip().upper()
                is_boilerplate = text_normalized in {p.upper() for p in PLATFORM_HEADINGS}

            # Also remove header/footer zone blocks from short documents less aggressively
            if block.block_type in ("header", "footer"):
                is_boilerplate = True

            if is_boilerplate:
                removed.append(block)
            else:
                kept.append(block)

        filtered[page_num] = kept

    logger.info("Removed %d boilerplate blocks, kept %d", len(removed), sum(len(v) for v in filtered.values()))
    return filtered, removed
