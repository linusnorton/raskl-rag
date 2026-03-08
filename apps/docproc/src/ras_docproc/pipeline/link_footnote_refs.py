"""Pipeline step: link in-body footnote references to footnote records."""

from __future__ import annotations

import logging
import re

from ras_docproc.pipeline.extract_mupdf import MuPDFPageData
from ras_docproc.schema import FootnoteRecord, FootnoteRefRecord, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Regex patterns for inline footnote references
REF_PATTERNS = [
    # Superscript-style: word followed by digit(s) then space or punctuation
    (re.compile(r"([A-Za-z\)])(\d{1,2})(\s|[.,;:])"), "regex_superscript"),
    # Period-separated: OCR renders superscript as "word.91" — period between letter and digits
    (re.compile(r"([A-Za-z])\.(\d{1,2})(\s|[A-Z]|[.,;:])"), "regex_period_superscript"),
    # Parenthesized: (1), (2), etc.
    (re.compile(r"\((\d{1,2})\)"), "regex_parens"),
    # Bracketed: [1], [2], etc.
    (re.compile(r"\[(\d{1,2})\]"), "regex_brackets"),
]


def link_footnote_refs(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    footnotes: list[FootnoteRecord],
    mupdf_data: dict[int, MuPDFPageData],
    doc_id: str,
) -> list[FootnoteRefRecord]:
    """Scan body blocks for footnote reference markers and link to FootnoteRecords.

    Uses two approaches:
    - Approach A: regex matching in text
    - Approach B: superscript span detection from MuPDF data

    Returns list of FootnoteRefRecord.
    """
    # Index footnotes by (page_num, footnote_number) for lookup
    fn_index: dict[tuple[int, int], FootnoteRecord] = {}
    for fn in footnotes:
        fn_index[(fn.page_num_1, fn.footnote_number)] = fn

    # Global footnote index by number only (for cross-page / endnote-style linking)
    fn_by_number: dict[int, FootnoteRecord] = {}
    for fn in footnotes:
        fn_by_number[fn.footnote_number] = fn
    all_fn_numbers = set(fn_by_number.keys())

    all_refs: list[FootnoteRefRecord] = []

    for page_num, blocks in blocks_by_page.items():
        page_fn_numbers = {fn.footnote_number for (pn, fn_num), fn in fn_index.items() if pn == page_num}

        body_blocks = [b for b in blocks if b.block_type not in ("footnote", "header", "footer")]

        for block in body_blocks:
            # Approach A: regex matching
            text = block.text_clean or block.text_raw
            for pattern, match_type in REF_PATTERNS:
                for m in pattern.finditer(text):
                    try:
                        if match_type in ("regex_superscript", "regex_period_superscript"):
                            num = int(m.group(2))
                        else:
                            num = int(m.group(1))
                    except (ValueError, IndexError):
                        continue

                    # Try same-page first, then fall back to global index
                    if num in page_fn_numbers:
                        fn = fn_index.get((page_num, num))
                        effective_match_type = match_type
                    elif num in all_fn_numbers:
                        fn = fn_by_number.get(num)
                        effective_match_type = match_type + "_xpage"
                    else:
                        continue

                    ref_id = make_block_id(
                        doc_id, page_num, f"ref-{block.block_id}", effective_match_type, text_hash(str(num))
                    )

                    # Context snippet
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    snippet = text[start:end]

                    all_refs.append(
                        FootnoteRefRecord(
                            ref_id=ref_id,
                            doc_id=doc_id,
                            page_num_1=page_num,
                            parent_block_id=block.block_id,
                            footnote_number=num,
                            footnote_id=fn.footnote_id if fn else None,
                            match_type=effective_match_type,
                            context_snippet=snippet,
                        )
                    )

            # Approach B: superscript span detection
            mupdf_page = mupdf_data.get(page_num)
            if mupdf_page:
                _find_superscript_refs(
                    block,
                    mupdf_page,
                    page_fn_numbers,
                    all_fn_numbers,
                    fn_index,
                    fn_by_number,
                    page_num,
                    doc_id,
                    all_refs,
                )

    # Deduplicate refs by (parent_block_id, footnote_number)
    seen: set[tuple[str, int]] = set()
    deduped: list[FootnoteRefRecord] = []
    for ref in all_refs:
        key = (ref.parent_block_id, ref.footnote_number)
        if key not in seen:
            seen.add(key)
            deduped.append(ref)

    logger.info("Linked %d footnote references", len(deduped))
    return deduped


def apply_ref_markup(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    footnote_refs: list[FootnoteRefRecord],
) -> dict[int, list[TextBlockRecord]]:
    """Post-process text_clean to insert [ref:N] markup for superscript footnote refs.

    For regex_superscript and superscript_span refs, replaces concatenated
    "word43" patterns with "word [ref:43]" in text_clean.

    Parenthesized/bracketed refs are left as-is since they're already readable.
    """
    # Match types that need [ref:N] markup inserted
    _MARKUP_MATCH_TYPES = {
        "regex_superscript",
        "regex_superscript_xpage",
        "regex_period_superscript",
        "regex_period_superscript_xpage",
        "superscript_span",
        "superscript_span_xpage",
    }

    # Index refs by parent block id
    refs_by_block: dict[str, list[FootnoteRefRecord]] = {}
    for ref in footnote_refs:
        if ref.match_type not in _MARKUP_MATCH_TYPES:
            continue
        refs_by_block.setdefault(ref.parent_block_id, []).append(ref)

    if not refs_by_block:
        return blocks_by_page

    modified = 0
    for page_num, blocks in blocks_by_page.items():
        for block in blocks:
            block_refs = refs_by_block.get(block.block_id)
            if not block_refs:
                continue

            text = block.text_clean
            # Sort refs by footnote number descending so replacements don't shift indices
            block_refs_sorted = sorted(block_refs, key=lambda r: r.footnote_number, reverse=True)
            for ref in block_refs_sorted:
                num_str = str(ref.footnote_number)
                base_type = ref.match_type.replace("_xpage", "")

                if base_type == "regex_period_superscript":
                    # Period-separated: "word.91" → "word [ref:91]"
                    pattern = re.compile(r"([A-Za-z])\." + re.escape(num_str) + r"(?=\s|[A-Z]|[.,;:]|$)")
                    replacement = rf"\1 [ref:{ref.footnote_number}]"
                else:
                    # Standard superscript: "word91" → "word [ref:91]"
                    pattern = re.compile(r"([A-Za-z\)])" + re.escape(num_str) + r"(?=\s|[.,;:]|$)")
                    replacement = rf"\1 [ref:{ref.footnote_number}]"

                new_text = pattern.sub(replacement, text)
                if new_text != text:
                    text = new_text
                    modified += 1

            block.text_clean = text

    logger.info("Applied [ref:N] markup to %d locations", modified)
    return blocks_by_page


def _find_superscript_refs(
    block: TextBlockRecord,
    mupdf_page: MuPDFPageData,
    page_fn_numbers: set[int],
    all_fn_numbers: set[int],
    fn_index: dict[tuple[int, int], FootnoteRecord],
    fn_by_number: dict[int, FootnoteRecord],
    page_num: int,
    doc_id: str,
    all_refs: list[FootnoteRefRecord],
) -> None:
    """Find superscript spans that overlap with this block's bbox."""
    if not mupdf_page.spans:
        return

    # Compute median font size for the page
    sizes = [s.font_size for s in mupdf_page.spans if s.font_size > 0]
    if not sizes:
        return
    sizes.sort()
    median_size = sizes[len(sizes) // 2]

    for span in mupdf_page.spans:
        # Superscript: significantly smaller font containing digits
        if span.font_size <= 0 or span.font_size >= 0.7 * median_size:
            continue

        text = span.text.strip()
        if not text.isdigit():
            continue

        num = int(text)

        # Try same-page first, then fall back to global index
        if num in page_fn_numbers:
            fn = fn_index.get((page_num, num))
            match_type = "superscript_span"
        elif num in all_fn_numbers:
            fn = fn_by_number.get(num)
            match_type = "superscript_span_xpage"
        else:
            continue

        # Check if span overlaps with block bbox (rough check)
        if (
            span.bbox.x0 >= block.bbox.x0 - 5
            and span.bbox.x1 <= block.bbox.x1 + 5
            and span.bbox.y0 >= block.bbox.y0 - 5
            and span.bbox.y1 <= block.bbox.y1 + 5
        ):
            ref_id = make_block_id(doc_id, page_num, f"ss-{block.block_id}", match_type, text_hash(str(num)))
            all_refs.append(
                FootnoteRefRecord(
                    ref_id=ref_id,
                    doc_id=doc_id,
                    page_num_1=page_num,
                    parent_block_id=block.block_id,
                    footnote_number=num,
                    footnote_id=fn.footnote_id if fn else None,
                    match_type=match_type,
                    evidence_span_bbox=span.bbox,
                )
            )
