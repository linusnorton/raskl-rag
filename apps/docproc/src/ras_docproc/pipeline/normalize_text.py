"""Pipeline step: normalize and clean text in blocks."""

from __future__ import annotations

import logging

from ras_docproc.schema import TextBlockRecord
from ras_docproc.utils.text import (
    clean_text,
    dehyphenate,
    normalize_superscript_refs,
    remove_script_intrusions,
    strip_diacritics,
)

logger = logging.getLogger(__name__)


def normalize_blocks(blocks_by_page: dict[int, list[TextBlockRecord]]) -> dict[int, list[TextBlockRecord]]:
    """Apply text normalization to all blocks, preserving text_raw.

    Sets text_clean on each block.
    """
    total = 0
    for page_num, blocks in blocks_by_page.items():
        for block in blocks:
            cleaned = clean_text(block.text_raw)
            cleaned = dehyphenate(cleaned)
            cleaned = normalize_superscript_refs(cleaned)
            block.text_clean = cleaned
            total += 1

    logger.info("Normalized text on %d blocks", total)
    return blocks_by_page


def ocr_cleanup_blocks(blocks_by_page: dict[int, list[TextBlockRecord]]) -> dict[int, list[TextBlockRecord]]:
    """Post-language-detection OCR cleanup.

    For English-detected blocks:
    - Strip spurious diacritics (OCR errors like "Apríl" -> "April")
    - Remove non-Latin script intrusions (Cyrillic, CJK fragments)

    Must be called AFTER language detection so we only modify lang="en" blocks.
    """
    cleaned = 0
    for page_num, blocks in blocks_by_page.items():
        for block in blocks:
            if block.lang != "en":
                continue

            original = block.text_clean
            text = strip_diacritics(original)
            text = remove_script_intrusions(text)

            if text != original:
                block.text_clean = text
                cleaned += 1

    logger.info("OCR cleanup: modified %d English blocks", cleaned)
    return blocks_by_page
