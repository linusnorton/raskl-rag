"""Pipeline step: extract document metadata from PDF metadata and cover page text."""

from __future__ import annotations

import logging
import re

from ras_docproc.schema import DocumentRecord, TextBlockRecord

logger = logging.getLogger(__name__)

# Regex for DOI
DOI_RE = re.compile(r"10\.\d{4,}/[^\s]+")

# Regex for stable URL
URL_RE = re.compile(r"https?://[^\s]+")

# Regex for JSTOR/academic source line, e.g.:
# "Source: Journal of ..., Vol. 84, No. 1 (June 2011), pp. 1-22"
SOURCE_RE = re.compile(
    r"Source:\s*(.+?)(?:,\s*Vol\.?\s*(\d+))?"
    r"(?:,\s*No\.?\s*(\d+))?"
    r"(?:\s*\(([^)]+)\))?"
    r"(?:,\s*pp?\.?\s*([\d]+\s*-\s*[\d]+))?"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Regex for MUSE citation format on cover pages. Handles both orderings of No./date:
#   "..., Volume 84, Part 1, June 2010, No. 300, pp. 1-22"
#   "..., Volume 93, Part 1, No. 318, June 2020, pp. 119-131"
MUSE_CITATION_RE = re.compile(
    r"(Journal of the Malaysian Branch of the Royal Asiatic Society)"
    r",\s*Volume\s*(\d+)"
    r"(?:,\s*Part\s*(\d+))?"
    r".*?,\s*pp?\.?\s*([\d]+\s*-\s*[\d]+)",
    re.IGNORECASE,
)

# Known-bad PDF metadata titles set by hosting platforms
BAD_TITLES = {"PROJECT MUSE", "JSTOR", "Untitled", "Layout 1"}

# Known platform headings to skip in title heuristic
PLATFORM_HEADINGS = {"PROJECT MUSE", "JSTOR"}

# Year extraction from various contexts
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")


def _clean_title(t: str) -> str:
    """Strip markdown formatting artifacts from title text."""
    t = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", t)  # **bold** / *italic*
    t = re.sub(r"^```(?:markdown)?\s*", "", t)  # code fence opener
    t = re.sub(r"```\s*$", "", t)  # code fence closer
    return t.strip()


def extract_metadata(
    document: DocumentRecord,
    blocks_by_page: dict[int, list[TextBlockRecord]],
    pdf_metadata: dict[str, str],
) -> DocumentRecord:
    """Extract metadata from PDF properties and cover page text.

    Reads PyMuPDF doc.metadata and parses first 2 pages for JSTOR/MUSE
    cover page patterns (Author(s):, Source:, Published by:, Stable URL:, DOI).

    Mutates and returns the DocumentRecord with populated fields.
    """
    # 1. Apply PDF-level metadata as defaults (skip known-bad platform titles)
    if pdf_metadata.get("title") and not document.title:
        candidate = pdf_metadata["title"].strip()
        candidate_normalized = re.sub(r"[^\w\s]", "", candidate).strip().upper()
        if candidate_normalized not in {re.sub(r"[^\w\s]", "", t).strip().upper() for t in BAD_TITLES}:
            document.title = candidate
    if pdf_metadata.get("author") and not document.author:
        document.author = pdf_metadata["author"].strip()

    # 2. Parse cover page text (first 2 pages)
    cover_lines: list[str] = []
    for page_num in sorted(blocks_by_page.keys())[:2]:
        for block in blocks_by_page[page_num]:
            text = block.text_clean or block.text_raw
            cover_lines.extend(text.splitlines())

    cover_text = "\n".join(cover_lines)

    # Author(s): line
    author_match = re.search(r"Author\(?s?\)?:\s*(.+)", cover_text, re.IGNORECASE)
    if author_match and not document.author:
        document.author = author_match.group(1).strip()

    # Source: line - parse journal, volume, issue, year, pages
    source_match = SOURCE_RE.search(cover_text)
    if source_match:
        journal = source_match.group(1)
        if journal and not document.publication:
            document.publication = journal.strip().rstrip(",")
        if source_match.group(2) and not document.volume:
            document.volume = source_match.group(2)
        if source_match.group(3) and not document.issue:
            document.issue = source_match.group(3)
        date_part = source_match.group(4)
        if date_part and not document.year:
            year_m = YEAR_RE.search(date_part)
            if year_m:
                document.year = int(year_m.group(1))
        if source_match.group(5) and not document.page_range_label:
            document.page_range_label = source_match.group(5)
        # Build journal_ref from parsed fields
        if not document.journal_ref and document.publication:
            parts = [document.publication]
            if document.volume:
                parts.append(f"Vol. {document.volume}")
            if document.issue:
                parts.append(f"No. {document.issue}")
            if document.page_range_label:
                parts.append(f"pp. {document.page_range_label}")
            document.journal_ref = ", ".join(parts)

    # MUSE citation format (if JSTOR Source: didn't match)
    if not source_match:
        muse_match = MUSE_CITATION_RE.search(cover_text)
        if muse_match:
            if muse_match.group(1) and not document.publication:
                document.publication = muse_match.group(1).strip()
            if muse_match.group(2) and not document.volume:
                document.volume = muse_match.group(2)
            if muse_match.group(3) and not document.issue:
                document.issue = muse_match.group(3)
            if muse_match.group(4) and not document.page_range_label:
                document.page_range_label = muse_match.group(4).replace(" ", "")
            # Extract year from the full MUSE citation line (e.g. "June 2012, No. 300")
            if not document.year:
                year_m = YEAR_RE.search(muse_match.group(0))
                if year_m:
                    document.year = int(year_m.group(1))
            # Build journal_ref
            if not document.journal_ref and document.publication:
                parts = [document.publication]
                if document.volume:
                    parts.append(f"Vol. {document.volume}")
                if document.issue:
                    parts.append(f"Part {document.issue}")
                if document.page_range_label:
                    parts.append(f"pp. {document.page_range_label}")
                document.journal_ref = ", ".join(parts)

    # Compute page_offset from page_range_label + page_count
    if document.page_range_label:
        try:
            end_page = int(document.page_range_label.split("-")[-1].strip())
            total_pages = int(pdf_metadata.get("page_count", 0))
            if total_pages > 0:
                document.page_offset = end_page - total_pages
        except (ValueError, IndexError):
            pass

    # Published by: line
    publisher_match = re.search(r"Published by:\s*(.+)", cover_text, re.IGNORECASE)
    if publisher_match and not document.publication:
        document.publication = publisher_match.group(1).strip()

    # Stable URL:
    url_line_match = re.search(r"Stable URL:\s*(.+)", cover_text, re.IGNORECASE)
    if url_line_match and not document.url:
        url_m = URL_RE.search(url_line_match.group(1))
        if url_m:
            document.url = url_m.group(0).rstrip(".")
    # Fallback: any URL on cover pages
    if not document.url:
        url_m = URL_RE.search(cover_text)
        if url_m:
            document.url = url_m.group(0).rstrip(".")

    # DOI
    doi_match = DOI_RE.search(cover_text)
    if doi_match and not document.doi:
        document.doi = doi_match.group(0).rstrip(".")

    # Title heuristic: largest heading on page 1, or first substantial line
    if not document.title:
        page1_blocks = blocks_by_page.get(1, []) or blocks_by_page.get(min(blocks_by_page.keys(), default=1), [])
        # Prefer heading blocks (skip platform headings like "PROJECT MUSE")
        headings = [b for b in page1_blocks if b.block_type == "heading"]
        first_is_platform = False
        if headings:
            for h in headings:
                text = (h.text_clean or h.text_raw).strip()
                text_normalized = re.sub(r"[^\w\s]", "", text).strip().upper()
                if text_normalized in {p.upper() for p in PLATFORM_HEADINGS}:
                    first_is_platform = True
                    continue
                if len(text) > 10:
                    document.title = _clean_title(text)
                    break
        # If the first heading was a platform name (e.g. "PROJECT MUSE"),
        # take the first paragraph after it as the title (not citation/boilerplate)
        if not document.title and first_is_platform:
            for b in page1_blocks:
                if b.block_type in ("header", "footer", "page_number", "heading"):
                    continue
                text = (b.text_clean or b.text_raw).strip()
                # Skip citation/boilerplate lines
                if MUSE_CITATION_RE.search(text) or SOURCE_RE.search(text):
                    continue
                if text.lower().startswith(("published by", "author", "source:", "stable url")):
                    continue
                if len(text) > 10:
                    document.title = _clean_title(text[:200])
                    break
        # Fallback: first substantial non-boilerplate paragraph
        if not document.title:
            for b in page1_blocks:
                if b.block_type in ("header", "footer", "page_number"):
                    continue
                text = (b.text_clean or b.text_raw).strip()
                if len(text) > 20:
                    document.title = _clean_title(text[:200])
                    break

    # Year fallback 1: extract from filename pattern "Author (YEAR) JMBRAS..."
    if not document.year:
        fn_year_m = re.search(r"\((\d{4})\)", document.source_filename or "")
        if fn_year_m:
            document.year = int(fn_year_m.group(1))

    # Year fallback 2: search cover text (risky — may find historical dates)
    if not document.year:
        year_m = YEAR_RE.search(cover_text)
        if year_m:
            document.year = int(year_m.group(1))

    # Final cleanup: strip markdown artifacts from title (Qwen3 VL output)
    if document.title:
        document.title = _clean_title(document.title)

    fields_found = sum(
        1 for f in [document.title, document.author, document.year, document.doi, document.url] if f is not None
    )
    logger.info("Metadata extraction: populated %d/5 key fields", fields_found)
    return document
