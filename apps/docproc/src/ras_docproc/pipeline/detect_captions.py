"""Pipeline step: match caption text blocks to figures."""

from __future__ import annotations

import logging
import re

from ras_docproc.schema import FigureRecord, PlateRecord, TextBlockRecord
from ras_docproc.utils.geometry import vertical_distance
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Patterns for figure/plate captions
CAPTION_PATTERNS = [
    re.compile(r"(?i)^fig(?:ure)?\.?\s*\d+"),
    re.compile(r"(?i)^plate\s+[IVXLCDM\d]+"),
    re.compile(r"(?i)^photo(?:graph)?\.?\s*\d+"),
    re.compile(r"(?i)^map\s+\d+"),
    re.compile(r"(?i)^illustration\s+\d+"),
]

PLATE_PATTERN = re.compile(r"(?i)plate\s+([IVXLCDM\d]+)")

MAX_CAPTION_DISTANCE = 50.0  # PDF points below figure


def detect_captions(
    figures: list[FigureRecord],
    blocks_by_page: dict[int, list[TextBlockRecord]],
    doc_id: str,
) -> tuple[list[FigureRecord], list[PlateRecord]]:
    """Match caption text blocks to figures, detect plate pages.

    For each figure bbox, search for text blocks below within threshold distance
    that match caption patterns. Also detect plate pages (multiple figures + "Plate" label).

    Returns:
        (updated figures, plate records)
    """
    plates: list[PlateRecord] = []

    # Group figures by page
    figs_by_page: dict[int, list[FigureRecord]] = {}
    for fig in figures:
        figs_by_page.setdefault(fig.page_num_1, []).append(fig)

    for page_num, page_figs in figs_by_page.items():
        blocks = blocks_by_page.get(page_num, [])
        if not blocks:
            continue

        # Sort figures by vertical position
        page_figs.sort(key=lambda f: f.bbox.y0 if f.bbox else 0)

        # Get image bboxes for boundary checking
        fig_bboxes = [f.bbox for f in page_figs if f.bbox]

        plate_label = None
        plate_caption_block_ids: list[str] = []

        for fig in page_figs:
            if not fig.bbox or fig.derived_from == "rendered_clip":
                continue

            caption_blocks: list[TextBlockRecord] = []
            for block in blocks:
                if block.block_type in ("footnote", "header", "footer"):
                    continue

                # Check if block is below the figure
                dist = vertical_distance(fig.bbox, block.bbox)
                if dist < 0 or dist > MAX_CAPTION_DISTANCE:
                    continue

                # Check horizontal overlap
                x_overlap = min(fig.bbox.x1, block.bbox.x1) - max(fig.bbox.x0, block.bbox.x0)
                if x_overlap < 0:
                    continue

                # Stop at next image bbox
                is_past_next_fig = False
                for other_bbox in fig_bboxes:
                    if other_bbox is not fig.bbox and block.bbox.y0 > other_bbox.y0 and other_bbox.y0 > fig.bbox.y1:
                        is_past_next_fig = True
                        break
                if is_past_next_fig:
                    break

                # Check if text matches caption pattern
                text = block.text_clean or block.text_raw
                is_caption = any(p.match(text.strip()) for p in CAPTION_PATTERNS)

                if is_caption or (caption_blocks and dist < 20):
                    caption_blocks.append(block)
                    block.block_type = "caption"

                    # Check for plate label
                    plate_match = PLATE_PATTERN.search(text)
                    if plate_match:
                        plate_label = plate_match.group(0)

            if caption_blocks:
                fig.caption_block_ids = [b.block_id for b in caption_blocks]
                fig.caption_text_raw = "\n".join(b.text_raw for b in caption_blocks)
                fig.caption_text_clean = "\n".join(b.text_clean or b.text_raw for b in caption_blocks)
                plate_caption_block_ids.extend(fig.caption_block_ids)

        # Detect plate page
        if plate_label and len(page_figs) >= 1:
            plate_id = make_block_id(doc_id, page_num, "plate", "plate", text_hash(plate_label))
            plates.append(
                PlateRecord(
                    plate_id=plate_id,
                    doc_id=doc_id,
                    page_num_1=page_num,
                    plate_label=plate_label,
                    figure_ids=[f.figure_id for f in page_figs],
                    plate_note_block_ids=plate_caption_block_ids,
                    plate_note_text_raw=plate_label,
                    plate_note_text_clean=plate_label,
                )
            )

    logger.info(
        "Matched captions for %d figures, found %d plates",
        sum(1 for f in figures if f.caption_block_ids),
        len(plates),
    )
    return figures, plates
