"""Citation extraction, renumbering, and formatting for RAG responses."""

from __future__ import annotations
import re
from .retriever import RetrievedChunk

_FILENAME_RE = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]')


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


def collapse_duplicate_indices(index_map: dict[int, int], all_chunks: list[RetrievedChunk]) -> dict[int, int]:
    """Merge indices that point to the same document and page into a single label."""
    if not index_map:
        return index_map

    chunk_key_for_orig = {}
    for orig in index_map:
        if 1 <= orig <= len(all_chunks):
            c = all_chunks[orig - 1]
            chunk_key_for_orig[orig] = (c.doc_id, c.start_page)

    key_to_display = {}
    collapsed = {}
    next_display = 1

    for orig in sorted(index_map, key=lambda o: index_map[o]):
        key = chunk_key_for_orig.get(orig)
        if key is None:
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


def _clean_metadata(text: str | None) -> str:
    """Removes messy markdown and standardises society/journal abbreviations."""
    if not text:
        return ""
    
    # 1. Strip corrupted asterisks and whitespace
    text = text.replace("*", "").strip()
    
    # 2. Priority Replacements (Longest/Most Specific First)
    replacements = [
        (r"Journal of the (Malaysian|Malayan) Branch of the Royal Asiatic Society", "JMBRAS"),
        (r"Journal of the Straits Branch of the Royal Asiatic Society", "JSBRAS"),
        (r"(Malaysian|Malayan) Branch of the Royal Asiatic Society", "MBRAS"),
        (r"Straits Branch of the Royal Asiatic Society", "JSBRAS"),
    ]
    
    for pattern, replacement in replacements:
        # Use case-insensitive regex to catch variations
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    return text

def _format_citation_line(label: int, c: RetrievedChunk, pages: str) -> str:
    """Format a citation in JMBRAS style: Author (Year). 'Title', *Publication*, Pages."""
    author = _clean_metadata(c.author or "")
    year = str(c.year) if c.year else ""
    title = _clean_metadata(c.title or c.source_filename)
    publication = _clean_metadata(c.publication)
    
    author_seg = f"{author} ({year})" if author and year else (author or (f"({year})" if year else ""))
    pages = pages.replace("-", "–") # JMBRAS requires en-dashes for ranges

    if publication:
        citation_body = f"'{title}', *{publication}*"
    else:
        citation_body = f"*{title}*"

    segments = [f"[{label}] {author_seg}".strip(), f"{citation_body}, {pages}" if pages else citation_body]
    
    if getattr(c, 'url', None):
        segments.append(f"🔗 {c.url}") # Need to add url column to documents table and update retriever.py

    return ". ".join(segments).replace("..", ".")

def format_citations(chunks: list[RetrievedChunk], response_text: str = "", index_map: dict[int, int] | None = None) -> str:
    """Build a deduplicated markdown citation block using JMBRAS formatting."""
    if not chunks:
        return ""

    cited = extract_cited_indices(response_text) if response_text else set()
    if response_text and not cited:
        return ""

    seen: set[str] = set()
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        if cited and i not in cited:
            continue

        key = (c.chunk_id, c.start_page)
        if key in seen:
            continue
        seen.add(key)

        label = index_map[i] if index_map and i in index_map else i

        # Standardize page range: single space after p./pp. and en-dash (–) for ranges [cite: 54, 68]
        display_start = c.start_page
        display_end = c.end_page
        if display_start == display_end:
            pages = f"p. {display_start}"
        else:
            pages = f"pp. {display_start}–{display_end}"

        # Call the updated 3-argument function
        line = _format_citation_line(label, c, pages)
        lines.append(line)

    if not lines:
        return ""
    lines.sort(key=lambda l: int(re.match(r'\[(\d+)\]', l).group(1)))
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def renumber_response(partial_text: str, all_chunks: list[RetrievedChunk]) -> str:
    """Main entry point: Renumbers text and appends the source bibliography."""
    # Strip LLM 'think' block and user-generated bibliographies
    content = re.sub(r'<think>.*?</think>\s*', '', partial_text, flags=re.DOTALL)
    content = re.sub(r'\n+(?:---\n)?\*{0,2}(?:Sources?|References|Bibliography):?.*', '', content, flags=re.IGNORECASE | re.DOTALL)

    mapping = make_index_map(content)
    mapping = collapse_duplicate_indices(mapping, all_chunks)
    
    if mapping:
        content = renumber_text(content, mapping)
    
    # Format the bibliography
    lines = []
    seen_keys = set()
    cited_indices = set(mapping.keys())
    
    for i, chunk in enumerate(all_chunks, start=1):
        if i not in cited_indices:
            continue
        key = (chunk.doc_id, chunk.start_page)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        
        display_label = mapping[i]
        pg_label = f"p. {chunk.start_page}" if chunk.start_page == chunk.end_page else f"pp. {chunk.start_page}–{chunk.end_page}"
        lines.append(_format_citation_line(display_label, chunk, pg_label))

    lines.sort(key=lambda x: int(re.match(r'\[(\d+)\]', x).group(1)))
    
    bibliography = "\n\n---\n**Sources:**\n" + "\n".join(lines) if lines else ""
    
    # Reconstruct final response with think block preserved
    think_match = re.match(r'<think>.*?</think>\s*', partial_text, re.DOTALL)
    prefix = think_match.group(0) if think_match else ""
    return prefix + content + bibliography