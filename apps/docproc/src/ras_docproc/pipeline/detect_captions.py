"""Pipeline step: match caption text blocks to figures."""

from __future__ import annotations

import logging
import re

from ras_docproc.pipeline.extract_mupdf import MuPDFPageData, SpanInfo
from ras_docproc.schema import BBox, FigureRecord, PlateRecord, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Patterns for figure/plate captions (used for plate detection, not as a gate for captions)
CAPTION_PATTERNS = [
    re.compile(r"(?i)^fig(?:ure)?\.?\s*\d+"),
    re.compile(r"(?i)^plate\s+[IVXLCDM\d]+"),
    re.compile(r"(?i)^photo(?:graph)?\.?\s*\d+"),
    re.compile(r"(?i)^map\s+\d+"),
    re.compile(r"(?i)^illustration\s+\d+"),
]

PLATE_PATTERN = re.compile(r"(?i)plate\s+([IVXLCDM\d]+)")

MAX_CAPTION_DISTANCE = 50.0  # PDF points below figure
MAX_CAPTION_LINES = 4  # Maximum number of text lines to collect as caption


def _group_spans_into_lines(spans: list[SpanInfo]) -> list[tuple[BBox, str]]:
    """Group nearby spans into text lines based on vertical proximity.

    Returns list of (line_bbox, line_text) sorted by y position.
    """
    if not spans:
        return []

    # Sort by vertical position then horizontal
    sorted_spans = sorted(spans, key=lambda s: (s.bbox.y0, s.bbox.x0))

    lines: list[tuple[BBox, list[str]]] = []
    current_line_y = sorted_spans[0].bbox.y0
    current_line_bbox = BBox(
        x0=sorted_spans[0].bbox.x0, y0=sorted_spans[0].bbox.y0,
        x1=sorted_spans[0].bbox.x1, y1=sorted_spans[0].bbox.y1,
    )
    current_line_texts: list[str] = [sorted_spans[0].text]

    for span in sorted_spans[1:]:
        # Same line if vertical overlap or very close
        if abs(span.bbox.y0 - current_line_y) < span.bbox.height * 0.5:
            current_line_bbox = BBox(
                x0=min(current_line_bbox.x0, span.bbox.x0),
                y0=min(current_line_bbox.y0, span.bbox.y0),
                x1=max(current_line_bbox.x1, span.bbox.x1),
                y1=max(current_line_bbox.y1, span.bbox.y1),
            )
            current_line_texts.append(span.text)
        else:
            lines.append((current_line_bbox, current_line_texts))
            current_line_y = span.bbox.y0
            current_line_bbox = BBox(
                x0=span.bbox.x0, y0=span.bbox.y0,
                x1=span.bbox.x1, y1=span.bbox.y1,
            )
            current_line_texts = [span.text]

    lines.append((current_line_bbox, current_line_texts))

    return [(bbox, " ".join(t.strip() for t in texts if t.strip())) for bbox, texts in lines]


def _find_caption_from_spans(
    fig: FigureRecord,
    page_data: MuPDFPageData,
    other_fig_tops: list[float],
) -> str:
    """Find caption text below a figure using MuPDF spans with accurate bboxes.

    Collects text lines directly below the figure within MAX_CAPTION_DISTANCE,
    stopping at the next figure or after MAX_CAPTION_LINES.
    """
    if not fig.bbox:
        return ""

    fig_bottom = fig.bbox.y1
    fig_x0 = fig.bbox.x0
    fig_x1 = fig.bbox.x1

    # Find the boundary: next figure top below this figure
    next_fig_top = float("inf")
    for top in other_fig_tops:
        if top > fig_bottom + 5:  # Must be meaningfully below
            next_fig_top = min(next_fig_top, top)

    # Collect spans below the figure within range
    candidate_spans: list[SpanInfo] = []
    for span in page_data.spans:
        span_top = span.bbox.y0
        # Must be below figure bottom (with small tolerance)
        if span_top < fig_bottom - 5:
            continue
        # Must be within caption distance
        if span_top > fig_bottom + MAX_CAPTION_DISTANCE:
            continue
        # Must not cross into next figure
        if span_top > next_fig_top - 5:
            continue
        # Must have horizontal overlap with figure
        x_overlap = min(fig_x1, span.bbox.x1) - max(fig_x0, span.bbox.x0)
        if x_overlap < 0:
            continue
        candidate_spans.append(span)

    if not candidate_spans:
        return ""

    # Group spans into lines
    lines = _group_spans_into_lines(candidate_spans)

    # Take up to MAX_CAPTION_LINES
    caption_lines = [text for _, text in lines[:MAX_CAPTION_LINES] if text.strip()]

    return "\n".join(caption_lines)


def detect_captions(
    figures: list[FigureRecord],
    blocks_by_page: dict[int, list[TextBlockRecord]],
    doc_id: str,
    mupdf_data: dict[int, MuPDFPageData] | None = None,
) -> tuple[list[FigureRecord], list[PlateRecord]]:
    """Match caption text to figures using MuPDF spans (accurate bboxes).

    Uses MuPDF span data for spatial matching since Qwen3 VL text blocks
    have degenerate full-page bboxes that make spatial matching impossible.

    Falls back to text block pattern matching when MuPDF data is unavailable.

    Returns:
        (updated figures, plate records)
    """
    mupdf_data = mupdf_data or {}
    plates: list[PlateRecord] = []

    # Group figures by page
    figs_by_page: dict[int, list[FigureRecord]] = {}
    for fig in figures:
        figs_by_page.setdefault(fig.page_num_1, []).append(fig)

    for page_num, page_figs in figs_by_page.items():
        page_data = mupdf_data.get(page_num)

        # Sort figures by vertical position
        page_figs.sort(key=lambda f: f.bbox.y0 if f.bbox else 0)

        # Get figure top positions for boundary checking
        fig_tops = [f.bbox.y0 for f in page_figs if f.bbox]

        plate_label = None

        for fig in page_figs:
            if not fig.bbox:
                continue
            # Skip figures that already have captions (e.g. VL-detected scans)
            if fig.caption_text_raw:
                continue
            if fig.derived_from in ("rendered_clip", "vl_detected_scan"):
                continue

            caption_text = ""

            # Primary: use MuPDF spans for spatial caption matching
            if page_data and page_data.spans:
                caption_text = _find_caption_from_spans(fig, page_data, fig_tops)

            # Fallback: try text block pattern matching (for blocks with real bboxes)
            if not caption_text:
                blocks = blocks_by_page.get(page_num, [])
                caption_text = _find_caption_from_blocks(fig, blocks, page_figs)

            if caption_text:
                fig.caption_text_raw = caption_text
                fig.caption_text_clean = caption_text

                # Check for plate label
                plate_match = PLATE_PATTERN.search(caption_text)
                if plate_match:
                    plate_label = plate_match.group(0)

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
                    plate_note_text_raw=plate_label,
                    plate_note_text_clean=plate_label,
                )
            )

    logger.info(
        "Matched captions for %d figures, found %d plates",
        sum(1 for f in figures if f.caption_text_raw),
        len(plates),
    )
    return figures, plates


def _find_caption_from_blocks(
    fig: FigureRecord,
    blocks: list[TextBlockRecord],
    page_figs: list[FigureRecord],
) -> str:
    """Legacy fallback: find caption using text block pattern matching.

    Only works when text blocks have real bboxes (not full-page degenerate ones).
    """
    from ras_docproc.utils.geometry import vertical_distance

    if not fig.bbox:
        return ""

    fig_bboxes = [f.bbox for f in page_figs if f.bbox]
    caption_texts: list[str] = []

    for block in blocks:
        if block.block_type in ("footnote", "header", "footer"):
            continue

        # Check if block bbox is degenerate (full-page) — skip if so
        if block.bbox.height > fig.bbox.height * 3:
            continue

        dist = vertical_distance(fig.bbox, block.bbox)
        if dist < 0 or dist > MAX_CAPTION_DISTANCE:
            continue

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

        text = block.text_clean or block.text_raw
        is_caption = any(p.match(text.strip()) for p in CAPTION_PATTERNS)

        if is_caption or (caption_texts and dist < 20):
            caption_texts.append(text)
            block.block_type = "caption"

    return "\n".join(caption_texts)
