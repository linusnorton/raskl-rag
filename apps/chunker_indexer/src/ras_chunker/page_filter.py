"""Filter bleed content from adjacent articles in journal PDFs.

Two levels of filtering:

Level 1 — Page-level: drop blocks outside the article's page range (from page_range_label).
Level 2 — Title-based: on the first page, find the title heading and drop preceding blocks
          that belong to the previous article.
"""

from __future__ import annotations

import re
import string

from .schema import StitchedBlock


def parse_page_range(page_range_label: str | None) -> tuple[int, int] | None:
    """Parse '57-61' into (57, 61). Returns None if unparseable."""
    if not page_range_label:
        return None
    m = re.match(r"(\d+)\s*[-–—]\s*(\d+)", page_range_label.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    # Single page: "57"
    m = re.match(r"(\d+)$", page_range_label.strip())
    if m:
        p = int(m.group(1))
        return p, p
    return None


def _normalize_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _title_matches(heading_text: str, title: str, threshold: float = 0.6) -> bool:
    """Check if heading matches title using normalized word overlap (≥threshold)."""
    heading_words = _normalize_words(heading_text)
    title_words = _normalize_words(title)
    if not title_words or not heading_words:
        return False
    overlap = len(heading_words & title_words)
    return overlap / len(title_words) >= threshold


def filter_blocks_by_article_range(
    blocks: list[StitchedBlock],
    page_range_label: str | None,
    title: str | None,
) -> list[StitchedBlock]:
    """Filter blocks to only those belonging to the target article.

    Level 1: Drop blocks outside [start, end] page range.
    Level 2: On the first page, find title heading and drop preceding blocks.
    """
    page_range = parse_page_range(page_range_label)
    if page_range is None:
        return blocks

    start, end = page_range

    # Level 1: page-level filtering
    filtered = [b for b in blocks if start <= b.start_page <= end]

    # Level 2: title-based trimming on first page
    if title and filtered:
        title_idx = None
        for i, block in enumerate(filtered):
            if block.start_page != start:
                break
            if block.block_type == "heading" and _title_matches(block.text, title):
                title_idx = i
                break

        if title_idx is not None and title_idx > 0:
            filtered = filtered[title_idx:]

    return filtered


def filter_footnotes_by_page_range(
    footnotes: list,
    page_range_label: str | None,
) -> list:
    """Filter footnote records to only those within the article's page range."""
    page_range = parse_page_range(page_range_label)
    if page_range is None:
        return footnotes
    start, end = page_range
    return [fn for fn in footnotes if start <= fn.page_num_1 <= end]
