"""Text normalization and cleaning utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """Clean text: NFKC normalize, remove soft hyphens and odd separators."""
    text = normalize_nfkc(text)
    # Remove soft hyphens
    text = text.replace("\u00ad", "")
    # Remove zero-width spaces and joiners
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    # Normalize various dash types to standard hyphen for consistency
    text = re.sub(r"[\u2010\u2011]", "-", text)
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Conservative line-break dehyphenation.

    Joins lines where a word is split with a hyphen at end of line,
    but only when the next line starts with a lowercase letter.
    """
    return re.sub(r"-\n([a-z])", r"\1", text)


def strip_diacritics(text: str) -> str:
    """NFD decompose, remove combining marks (category 'Mn').

    Converts accented characters to their base form: "Apríl" -> "April", "café" -> "cafe".
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def remove_script_intrusions(text: str) -> str:
    """Remove non-Latin script characters that are likely OCR errors.

    Strips Cyrillic (U+0400-04FF), CJK unified ideographs, and other
    non-Latin blocks from text expected to be Latin script.
    Preserves ASCII, Latin Extended, punctuation, digits, and whitespace.
    """
    # Remove Cyrillic block
    text = re.sub(r"[\u0400-\u04FF]+", "", text)
    # Remove CJK unified ideographs
    text = re.sub(r"[\u4E00-\u9FFF]+", "", text)
    # Remove CJK compatibility ideographs
    text = re.sub(r"[\uF900-\uFAFF]+", "", text)
    # Remove Arabic block (only when surrounded by Latin context)
    text = re.sub(r"[\u0600-\u06FF]+", "", text)
    # Collapse any resulting double spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def normalize_superscript_refs(text: str) -> str:
    """Convert HTML/LaTeX superscript footnote refs to parenthesized form.

    - ``<sup>17</sup>`` → ``(17)``
    - ``<sup>3a</sup>`` → ``(3a)``
    - ``\\(^{91}\\)`` → ``(91)``
    """
    text = re.sub(r"<sup>(\d+[a-z]?)</sup>", r"(\1)", text)
    text = re.sub(r"\\\(\^\{(\d+[a-z]?)\}\\\)", r"(\1)", text)
    return text


def normalize_for_frequency(line: str) -> str:
    """Normalize a line for frequency counting (boilerplate detection).

    Casefold, collapse whitespace, strip page numbers.
    """
    line = line.casefold()
    # Strip leading/trailing page numbers
    line = re.sub(r"^\d+\s*", "", line)
    line = re.sub(r"\s*\d+$", "", line)
    # Collapse whitespace
    line = re.sub(r"\s+", " ", line)
    return line.strip()
