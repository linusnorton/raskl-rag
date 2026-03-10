"""Pipeline step: detect footnote blocks in the page footer zone."""

from __future__ import annotations

import logging
import re
from typing import Literal

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, FootnoteRecord, TextBlockRecord
from ras_docproc.utils.geometry import is_in_zone
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Patterns for footnote number at start of text
FOOTNOTE_PATTERNS = [
    re.compile(r"^(\d{1,3})\s"),  # "1 Some footnote text"
    re.compile(r"^\((\d{1,3})\)\s*"),  # "(1) Some footnote text"
    re.compile(r"^(\d{1,3})\.\s"),  # "1. Some footnote text"
    re.compile(r"^(\d{1,3})\)[\s]"),  # "1) Some footnote text"
]

# ---------------------------------------------------------------------------
# Footnote type classification
# ---------------------------------------------------------------------------

_CITATION_PATTERNS = [
    # Scholarly author-year: "Gullick (1992: 246-8)", "Wong (2015)", "Zedler (1731-54)"
    re.compile(r"\b[A-Z][a-z]+\s*\(\d{4}(?:[a-z])?(?::\s*[\d\-–—,\s]+)?\)"),
    # Archival / manuscript sources
    re.compile(r"\bCO\s*\d+/\d+"),
    re.compile(r"\bFO\s*\d+/\d+"),
    re.compile(r"\b(?:SSF|SSFR)\b"),
    re.compile(r"\bBL/APAC/"),
    re.compile(r"\bIOR\b"),
    re.compile(r"\bHSL\s*\d+\.\d+"),
    # Ibid / cross-references
    re.compile(r"\bIbid\.?\b", re.IGNORECASE),
    re.compile(r"\bop\.\s*cit\.?\b", re.IGNORECASE),
    re.compile(r"\bloc\.\s*cit\.?\b", re.IGNORECASE),
    re.compile(r"\bNote\s+\d+\s+above\b", re.IGNORECASE),
    # See / cf. references
    re.compile(r"\bSee\s+(?:also\s+)?(?:fn\.?\s*\d+|[A-Z][a-z]+)"),
    re.compile(r"\bcf\.?\s+[A-Z]"),
    # Newspaper / periodical citations
    re.compile(r"\b(?:Malaya Tribune|Straits Times|Nanyang Siang Pau|The Sun)\b", re.IGNORECASE),
    re.compile(r",\s*\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"),
    # Official documents / dispatches
    re.compile(r"\bSSD\s*,"),
    re.compile(r"\bSSJ\s*,"),
    re.compile(r"['\u2019]s\s+(?:journal|diary)\b"),
    re.compile(r"Annual\s+Report\s+\d{4}"),
    re.compile(r"National\s+Archives\b"),
    # URLs / internet sources
    re.compile(r"https?://"),
    re.compile(r"\baccessed\s+\d{1,2}\s+\w+\s+\d{4}"),
    # Personal communications
    re.compile(r"\bpersonal\s+communication\b", re.IGNORECASE),
]


def classify_footnote_type(text: str) -> Literal["citation", "explanatory", "mixed"]:
    """Classify a footnote as citation, explanatory, or mixed.

    - ``citation``: the footnote is primarily a bibliographic reference
    - ``explanatory``: the footnote provides additional explanation with no external source
    - ``mixed``: the footnote cites a source *and* includes substantial explanatory content
    """
    # Strip footnote number prefix (e.g. "35 Weld to..." -> "Weld to...")
    stripped = re.sub(r"^\d{1,3}[\s.)]+", "", text.strip())

    # Check for citation patterns
    has_citation = any(p.search(stripped) for p in _CITATION_PATTERNS)
    if not has_citation:
        return "explanatory"

    # Determine if there's substantial explanatory content beyond the citation
    remaining = stripped
    for p in _CITATION_PATTERNS:
        remaining = p.sub("", remaining)
    # Strip punctuation, whitespace, and short connecting words
    remaining = re.sub(r"[\s.,;:()\[\]'\"-]+", " ", remaining).strip()
    remaining_words = len(remaining.split()) if remaining else 0

    if remaining_words <= 8:
        return "citation"
    return "mixed"


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
                # Already classified by extraction stage
                text = block.text_clean or block.text_raw
                _emit_footnotes(text, block, page_num, doc_id, all_footnotes)
                continue

            # Check if block is in footnote zone
            in_zone = is_in_zone(block.bbox, page_h, config.footnote_zone_top, config.footnote_zone_bottom)

            # Fallback: when bbox spans the full page (degenerate Qwen3 VL bbox),
            # zone check is meaningless — classify purely by text pattern.
            text_fallback = not in_zone and block.block_type == "paragraph" and _is_full_page_bbox(block.bbox, page_h)

            if not in_zone and not text_fallback:
                continue

            text = block.text_clean or block.text_raw
            fn_num = _extract_footnote_number(text)
            if fn_num is not None:
                block.block_type = "footnote"
                _emit_footnotes(text, block, page_num, doc_id, all_footnotes)

    logger.info("Detected %d footnotes across %d pages", len(all_footnotes), len(blocks_by_page))
    return blocks_by_page, all_footnotes


def _emit_footnotes(
    text: str,
    block: TextBlockRecord,
    page_num: int,
    doc_id: str,
    all_footnotes: list[FootnoteRecord],
) -> None:
    """Create FootnoteRecord(s) from a footnote block, splitting merged footnotes."""
    # Try splitting multi-footnote blocks first
    splits = _split_multi_footnote(text)
    if splits:
        for fn_num, fn_text in splits:
            # Reconstruct the footnote text with number prefix for classification
            full_text = f"{fn_num} {fn_text}"
            fn_id = make_block_id(doc_id, page_num, "fn", "footnote", text_hash(full_text))
            fn_type = classify_footnote_type(full_text)
            all_footnotes.append(
                FootnoteRecord(
                    footnote_id=fn_id,
                    doc_id=doc_id,
                    page_num_1=page_num,
                    footnote_number=fn_num,
                    bbox=block.bbox,
                    text_raw=full_text,
                    text_clean=full_text,
                    footnote_type=fn_type,
                )
            )
        return

    # Single footnote
    fn_num = _extract_footnote_number(text)
    if fn_num is not None:
        fn_id = make_block_id(doc_id, page_num, "fn", "footnote", text_hash(block.text_raw))
        fn_type = classify_footnote_type(text)
        all_footnotes.append(
            FootnoteRecord(
                footnote_id=fn_id,
                doc_id=doc_id,
                page_num_1=page_num,
                footnote_number=fn_num,
                bbox=block.bbox,
                text_raw=block.text_raw,
                text_clean=block.text_clean,
                footnote_type=fn_type,
            )
        )


def _extract_footnote_number(text: str) -> int | None:
    """Try to extract a footnote number from the start of text."""
    text = text.strip()
    for pattern in FOOTNOTE_PATTERNS:
        m = pattern.match(text)
        if m:
            return int(m.group(1))
    return None


# Regex to split merged footnotes: "35 Text here. 36 More text." → [(35, "Text here."), (36, "More text.")]
_MULTI_FN_SPLIT = re.compile(r"(?:^|\.\s+)(\d{1,3})(?:\s|$)")


def _split_multi_footnote(text: str) -> list[tuple[int, str]]:
    """Split a block that may contain multiple footnotes into (number, text) pairs.

    Qwen3 VL often merges consecutive footnotes into a single text block, e.g.:
    "35 Weld to Kimberley, CO 273/105. 36 Smith (1990: 12)."
    → [(35, "Weld to Kimberley, CO 273/105."), (36, "Smith (1990: 12).")]
    """
    text = text.strip()

    # Find all footnote number positions
    positions: list[tuple[int, int]] = []  # (char_offset, footnote_number)
    for m in _MULTI_FN_SPLIT.finditer(text):
        num = int(m.group(1))
        # Use the start of the digit, not the full match (which may include ". ")
        positions.append((m.start(1), num))

    if len(positions) <= 1:
        # Single footnote — use the normal extraction path
        return []

    # Validate: footnote numbers should be ascending and sequential (or nearly so)
    numbers = [n for _, n in positions]
    if not all(numbers[i] < numbers[i + 1] for i in range(len(numbers) - 1)):
        return []
    if numbers[-1] - numbers[0] >= len(numbers) * 3:
        # Numbers too spread out — probably not merged footnotes
        return []

    # Split text at each footnote number boundary
    result: list[tuple[int, str]] = []
    for i, (offset, num) in enumerate(positions):
        # Text starts after "N " (number + space)
        num_str = str(num)
        text_start = offset + len(num_str)
        # Text ends at the next footnote's ". N" boundary or end of string
        if i + 1 < len(positions):
            next_offset = positions[i + 1][0]
            # Walk back to find the ". " before the next number
            fn_text = text[text_start:next_offset].strip()
            # Remove trailing period/space if present (it belongs to the boundary)
            fn_text = fn_text.rstrip()
        else:
            fn_text = text[text_start:].strip()
        result.append((num, fn_text))

    return result
