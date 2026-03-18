"""Citation extraction, renumbering, and formatting for RAG responses."""

from __future__ import annotations

import re

from .retriever import RetrievedChunk

_FILENAME_RE = re.compile(r"^(.+?)\s*\((\d{4})\)")


def _parse_filename_metadata(filename: str) -> tuple[str, str]:
    """Extract author and year from source_filename like 'Abdullah (2011) JMBRAS 84(1), 1-22.pdf'."""
    m = _FILENAME_RE.match(filename)
    if m:
        return m.group(1).strip(), m.group(2)
    return "", ""

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


# Patterns that signal an LLM-generated sources/references/bibliography section.
# Matches optional leading `---\n` + optional bold `**` + keyword + optional bold + colon + newline.
_LLM_SOURCES_RE = re.compile(
    r'\n+(?:---\n)?'
    r'\*{0,2}(?:Sources?|References|Bibliography):?\*{0,2}:?\s*\n',
    re.IGNORECASE,
)


def strip_llm_sources(text: str) -> str:
    """Remove any LLM-generated Sources/References/Bibliography section from the end of the response.

    This is a safety net for when the LLM ignores the system prompt instruction not to generate
    a bibliography. The code-generated Sources block is appended separately by format_citations().
    Only strips if the match is past the first 30% of the text to avoid false positives
    (e.g. a sentence like "The sources indicate that...").
    """
    m = _LLM_SOURCES_RE.search(text)
    if not m:
        return text
    # Only strip if the matched section is past the first 30% of the text
    if m.start() < len(text) * 0.3:
        return text
    return text[:m.start()].rstrip()


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


def collapse_duplicate_indices(
    index_map: dict[int, int],
    all_chunks: list[RetrievedChunk],
) -> dict[int, int]:
    """Collapse indices that point to the same (chunk_id, start_page) to the same display number.

    When multiple context passages come from the same underlying chunk, the LLM may cite them
    with different [N] markers. This function merges them so they share a single display number,
    then renumbers the remaining display numbers consecutively.
    """
    if not index_map:
        return index_map

    # Map each original index to its chunk identity
    chunk_key_for_orig: dict[int, tuple[str, int]] = {}
    for orig in index_map:
        if 1 <= orig <= len(all_chunks):
            c = all_chunks[orig - 1]
            chunk_key_for_orig[orig] = (c.chunk_id, c.start_page)

    # Group original indices by chunk identity, assign first-seen display number
    key_to_display: dict[tuple[str, int], int] = {}
    collapsed: dict[int, int] = {}
    next_display = 1

    # Iterate in display-number order (i.e. order of first appearance in text)
    for orig in sorted(index_map, key=lambda o: index_map[o]):
        key = chunk_key_for_orig.get(orig)
        if key is None:
            # Index doesn't map to a known chunk — keep its display number
            collapsed[orig] = next_display
            next_display += 1
            continue
        if key not in key_to_display:
            key_to_display[key] = next_display
            next_display += 1
        collapsed[orig] = key_to_display[key]

    return collapsed


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


def _format_citation_line(label: int, c: RetrievedChunk, pages: str) -> str:
    """Format a single citation in academic style.

    Target: [N] Author (year). Title. Publication, pages · Type
    Fields are omitted when not available.
    """
    segments: list[str] = []

    # Use DB metadata, falling back to parsing the source filename
    fn_author, fn_year = _parse_filename_metadata(c.source_filename)
    author = c.author or fn_author
    year = fn_year or (str(c.year) if c.year else "")

    # Author (year)
    if author:
        segments.append(f"{author} ({year})" if year else author)
    elif year:
        segments.append(f"({year})")

    # Title — italicised
    title = c.title or c.source_filename
    segments.append(f"*{title}*")

    # Publication: Publisher, pages
    if c.publication:
        segments.append(f"{c.publication}, {pages}")
    else:
        segments.append(pages)

    return f"[{label}] " + ". ".join(segments)


def format_citations(chunks: list[RetrievedChunk], response_text: str = "", index_map: dict[int, int] | None = None) -> str:
    """Build a deduplicated markdown citation block from retrieved chunks.

    Only includes chunks that the LLM actually cited via [N] markers.
    If the response contains text but no [N] markers, returns no sources.
    index_map remaps original chunk indices to consecutive display labels.
    """
    if not chunks:
        return ""

    cited = extract_cited_indices(response_text) if response_text else set()

    # If the LLM produced a response but cited nothing, show no sources
    # (avoids dumping all retrieved chunks for pure-image or no-citation responses)
    if response_text and not cited:
        return ""

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

        display_start = c.start_page
        display_end = c.end_page
        pages = f"p.{display_start}" if display_start == display_end else f"pp.{display_start}-{display_end}"

        line = _format_citation_line(label, c, pages)
        lines.append(line)

    if not lines:
        return ""
    # Sort by display label so sources appear in [1], [2], [3] order
    lines.sort(key=lambda l: int(re.match(r'\[(\d+)\]', l).group(1)))
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def renumber_response(partial_text: str, all_chunks: list[RetrievedChunk]) -> str:
    """Renumber citations in a complete response and append formatted sources."""
    content = extract_content(partial_text)
    # Strip any LLM-generated sources section before processing citations
    content = strip_llm_sources(content)
    mapping = make_index_map(content)
    # Collapse duplicate chunk references to the same display number
    mapping = collapse_duplicate_indices(mapping, all_chunks)
    if mapping:
        renumbered = renumber_text(content, mapping)
        think_match = re.match(r'<think>.*?</think>\s*', partial_text, re.DOTALL)
        partial_text = (think_match.group(0) if think_match else '') + renumbered
    else:
        # Still need to apply stripped content (no citations, but sources may have been stripped)
        think_match = re.match(r'<think>.*?</think>\s*', partial_text, re.DOTALL)
        partial_text = (think_match.group(0) if think_match else '') + content
    # Pass original content (pre-renumber) so cited filter uses original LLM chunk indices
    citations = format_citations(all_chunks, content, index_map=mapping)
    if citations and partial_text:
        return partial_text + citations
    return partial_text
