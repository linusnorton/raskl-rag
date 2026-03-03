"""Pipeline step: detect language per text block."""

from __future__ import annotations

import logging

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import TextBlockRecord

logger = logging.getLogger(__name__)

# Map lingua language names to ISO codes
LINGUA_LANG_MAP = {
    "ENGLISH": "en",
    "MALAY": "ms",
    "CHINESE": "zh",
    "ARABIC": "ar",
    "FRENCH": "fr",
    "GERMAN": "de",
    "DUTCH": "nl",
    "PORTUGUESE": "pt",
    "SPANISH": "es",
    "ITALIAN": "it",
    "LATIN": "la",
}

# Map our config lang codes to lingua Language enum names
LANG_CODE_TO_LINGUA = {
    "en": "ENGLISH",
    "ms": "MALAY",
    "zh": "CHINESE",
    "ar": "ARABIC",
    "fr": "FRENCH",
    "de": "GERMAN",
    "nl": "DUTCH",
    "pt": "PORTUGUESE",
    "es": "SPANISH",
    "it": "ITALIAN",
    "la": "LATIN",
}


def _build_detector(lang_set: list[str]):
    """Build a lingua LanguageDetector for the specified languages."""
    from lingua import Language, LanguageDetectorBuilder

    languages = []
    for code in lang_set:
        lingua_name = LANG_CODE_TO_LINGUA.get(code)
        if lingua_name:
            lang_enum = getattr(Language, lingua_name, None)
            if lang_enum:
                languages.append(lang_enum)

    if not languages:
        languages = [Language.ENGLISH]

    return LanguageDetectorBuilder.from_languages(*languages).build()


def detect_languages(
    blocks_by_page: dict[int, list[TextBlockRecord]],
    config: PipelineConfig,
) -> dict[int, list[TextBlockRecord]]:
    """Detect language for each text block.

    Blocks with fewer than min_lang_chars characters get lang='unknown'.
    """
    detector = _build_detector(config.lang_set)
    total = 0
    detected = 0

    for page_num, blocks in blocks_by_page.items():
        for block in blocks:
            total += 1
            text = block.text_clean or block.text_raw
            if len(text) < config.min_lang_chars:
                block.lang = "unknown"
                block.lang_confidence = 0.0
                continue

            confidence_values = detector.compute_language_confidence_values(text)
            if confidence_values:
                top = confidence_values[0]
                lang_name = top.language.name
                block.lang = LINGUA_LANG_MAP.get(lang_name, lang_name.lower()[:2])
                block.lang_confidence = round(top.value, 4)

                # Top 3 candidates
                block.lang_candidates = []
                for cv in confidence_values[:3]:
                    code = LINGUA_LANG_MAP.get(cv.language.name, cv.language.name.lower()[:2])
                    block.lang_candidates.append(code)
                detected += 1
            else:
                block.lang = "unknown"
                block.lang_confidence = 0.0

    logger.info("Language detected for %d/%d blocks", detected, total)
    return blocks_by_page
