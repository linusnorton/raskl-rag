"""Citation extraction, renumbering, and formatting for RAG responses."""

from __future__ import annotations

import re

from .retriever import RetrievedChunk

# Matches both single [N] and multi-citation [N, M, ...] patterns
_CITATION_RE = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')


def _parse_indices(match_text: str) -> list[int]:
    """Parse comma-separated indices from a citation match group."""
    return [int(x.strip()) for x in match_text.split(",")]


def extract_cited_indices(text: str) -> set[int]:
    """Extract citation indices from LLM response text.

    Handles both [N] and [N, M] style citations.
    """
    indices: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        indices.update(_parse_indices(m.group(1)))
    return indices


def extract_content(text: str) -> str:
    """Strip <think>...</think> block and return only the final response body."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


def make_index_map(text: str) -> dict[int, int]:
    """Map original [N] citation indices to consecutive display numbers (in order of first appearance).

    Handles both [N] and [N, M] style citations.
    """
    seen: dict[int, int] = {}
    for m in _CITATION_RE.finditer(text):
        for orig in _parse_indices(m.group(1)):
            if orig not in seen:
                seen[orig] = len(seen) + 1
    return seen


def renumber_text(text: str, mapping: dict[int, int]) -> str:
    """Replace citation markers using consecutive display numbers.

    Handles both [N] and [N, M] patterns. Uses a sentinel to avoid cascading replacements.
    """
    def _replace(m: re.Match) -> str:
        indices = _parse_indices(m.group(1))
        renumbered = [str(mapping.get(i, i)) for i in indices]
        return "[\x00" + ", ".join(renumbered) + "\x00]"

    text = _CITATION_RE.sub(_replace, text)
    return text.replace("[\x00", "[").replace("\x00]", "]")


def format_citations(chunks: list[RetrievedChunk], response_text: str = "", index_map: dict[int, int] | None = None) -> str:
    """Build a deduplicated markdown citation block from retrieved chunks.

    Only includes chunks that the LLM actually cited via [N] markers.
    If no cited indices are found, falls back to showing all chunks.
    index_map remaps original chunk indices to consecutive display labels.
    """
    if not chunks:
        return ""

    cited = extract_cited_indices(response_text) if response_text else set()

    seen: set[str] = set()
    lines: list[str] = []
    display_n = 0
    for i, c in enumerate(chunks, start=1):
        # Filter to only cited chunks (if we detected any citations)
        if cited and i not in cited:
            continue

        key = (c.chunk_id, c.start_page)
        if key in seen:
            continue
        seen.add(key)

        display_n += 1
        label = index_map[i] if index_map and i in index_map else display_n

        source = c.title or c.source_filename
        display_start = c.start_page
        display_end = c.end_page
        pages = f"p.{display_start}" if display_start == display_end else f"pp.{display_start}-{display_end}"
        heading = f" — {c.section_heading}" if c.section_heading else ""
        author_year = ""
        if c.author or c.year:
            parts = []
            if c.author:
                parts.append(c.author)
            if c.year:
                parts.append(str(c.year))
            author_year = f" ({', '.join(parts)})"
        lines.append(f"[{label}] {source}{author_year}, {pages}{heading}")

    if not lines:
        return ""
    # Sort by display label so sources appear in [1], [2], [3] order
    lines.sort(key=lambda l: int(re.match(r'\[(\d+)\]', l).group(1)))
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def renumber_response(partial_text: str, all_chunks: list[RetrievedChunk]) -> str:
    """Renumber citations in a complete response and append formatted sources."""
    content = extract_content(partial_text)
    mapping = make_index_map(content)
    if mapping:
        renumbered = renumber_text(content, mapping)
        think_match = re.match(r'<think>.*?</think>\s*', partial_text, re.DOTALL)
        partial_text = (think_match.group(0) if think_match else '') + renumbered
    # Pass original content (pre-renumber) so cited filter uses original LLM chunk indices
    citations = format_citations(all_chunks, content, index_map=mapping)
    if citations and partial_text:
        return partial_text + citations
    return partial_text
