from __future__ import annotations
import re

# Publication name aliases for ILIKE matching (the publication field is inconsistent)
_PUBLICATION_PATTERNS = {
    "JMBRAS": "%Malay% Branch of the Royal Asiatic Society%",
    "MBRAS": "%Malay% Branch of the Royal Asiatic Society%",
    "JSBRAS": "%Straits Branch of the Royal Asiatic Society%",
    "SB": "%Straits Branch of the Royal Asiatic Society%",
}

_TYPE_LABELS = {
    "journal_article": " [Journal Article]",
    "primary_source": " [Primary Source]",
    "secondary_source": " [Secondary Source]",
    "front_matter": " [Front Matter]",
    "obituary": " [Obituary]",
    "editors_note": " [Editor's Note]",
    "illustration": " [Illustration]",
    "annual_report": " [Annual Report]",
    "agm_minutes": " [AGM Minutes]",
    "biographical_notes": " [Biographical Notes]",
    "mbras_monograph": " [MBRAS Monograph]",
    "mbras_reprint": " [MBRAS Reprint]",
    "index": " [Index]",
}

def resolve_publication_pattern(value: str | None) -> str | None:
    """Convert a publication filter to an ILIKE pattern."""
    if not value:
        return None
    return _PUBLICATION_PATTERNS.get(value.upper(), f"%{value}%")

def resolve_author_pattern(author: str | None) -> list[str]:
    """
    Split an author name into parts to handle format variations 
    (e.g., 'Azmi Khalid, A.' vs 'A. AZMI ABDUL KHALID').
    """
    if not author:
        return []
    # Split by comma or space, and filter out single characters or initials for the primary match
    return [p.strip() for p in author.replace(',', ' ').split() if len(p.strip()) > 1]
    # apps/rag_engine/src/ras_rag_engine/tools/utils.py

def get_type_label(doc_type: str | None) -> str:
    """Helper to safely get a formatted label for a document type."""
    if not doc_type:
        return ""
    return _TYPE_LABELS.get(doc_type, f" [{doc_type}]")

def parse_filename_metadata(filename: str) -> tuple[str, str]:
    """Extract author and year from source_filename like 'Author (Year) ...'."""
    m = re.match(r"^(.+?)\s*\((\d{4})\)", filename)
    if m:
        return m.group(1).strip(), m.group(2)
    return "", ""