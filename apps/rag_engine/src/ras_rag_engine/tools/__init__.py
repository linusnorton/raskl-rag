from __future__ import annotations
import re
import json
from . import search_basic, search_attribute, figures, browse, keyword_search, document_context, mbras_index
from ..retriever import RetrievedChunk
from .utils import get_type_label, parse_filename_metadata

# 1. Expose the definitions for the Agent's system prompt
def get_tool_definitions(config):
    return [
        search_basic.DEFINITION,
        search_attribute.DEFINITION,
        figures.DEFINITION,
        browse.DEFINITION,
        keyword_search.DEFINITION,
        document_context.DEFINITION,
        mbras_index.DEFINITION
    ]

# 2. Central Registry for dispatching calls
_TOOL_MAP = {
    "search_documents": search_basic.execute,
    "search_by_attribute": search_attribute.execute,
    "find_images": figures.execute,
    "browse_corpus": browse.execute,
    "keyword_search": keyword_search.execute,
    "document_context": document_context.execute,
    "mbras_index": mbras_index.execute
}

def execute_tool_call(name: str, args_json: str, config, start_index: int = 1):
    """Dispatches tool calls to the correct module logic."""
    args = json.loads(args_json) if isinstance(args_json, str) else args_json
    executor = _TOOL_MAP.get(name)
    
    if not executor:
        raise ValueError(f"Unknown tool: {name}")
        
    return executor(args, config, start_index=start_index)

# 3. Helper for context formatting (Moved from tools.py)
def format_chunks_for_context(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks for inclusion in the system prompt."""
    if not chunks:
        return "No documents retrieved yet."
    
    lines = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] SOURCE: {c.source_filename}"
        if c.title:
            header += f" | TITLE: {c.title}"
        if c.author:
            header += f" | AUTHOR: {c.author}"
        lines.append(f"{header}\n{c.text}\n---")
        
    return "\n\n".join(lines)

def format_chunks_for_context(
    chunks: list[RetrievedChunk],
    start_index: int = 1,
    figures: list[RetrievedFigure] | None = None,
) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM."""
    if not chunks:
        return "(No relevant passages found.)"

    # Build page→figures lookup
    page_figures: dict[tuple[str, int], list[RetrievedFigure]] = {}
    for fig in figures or []:
        page_figures.setdefault((fig.doc_id, fig.page_num), []).append(fig)

    parts = []
    for i, c in enumerate(chunks, start=start_index):
        source = c.title or c.source_filename
        display_start = c.start_page
        display_end = c.end_page
        pages = f"p.{display_start}" if display_start == display_end else f"pp.{display_start}-{display_end}"
        heading = f" — {c.section_heading}" if c.section_heading else ""
        # Document type label
        type_label = get_type_label(c.document_type)
        # Use DB metadata, falling back to parsing the source filename
        if c.source_filename:
            fn_author, fn_year = parse_filename_metadata(c.source_filename)
        else:
            fn_author, fn_year = None, None
        author = c.author or fn_author
        year = fn_year or (str(c.year) if c.year else "")
        # Build citation-style header: Author (Year), "Title", pages
        if author and year:
            header = f"[{i}]{type_label} {author} ({year}), \"{source}\", {pages}{heading}"
        elif author:
            header = f"[{i}]{type_label} {author}, \"{source}\", {pages}{heading}"
        elif year:
            header = f"[{i}]{type_label} \"{source}\" ({year}), {pages}{heading}"
        else:
            header = f"[{i}]{type_label} {source}, {pages}{heading}"

        chunk_text = f"{header}\n{c.text}"

        # Append contextual figures on the same page(s)
        seen_figs: set[str] = set()
        for p in range(c.start_page, c.end_page + 1):
            for fig in page_figures.get((c.doc_id, p), []):
                if fig.figure_id not in seen_figs:
                    seen_figs.add(fig.figure_id)
                    caption = fig.caption or ""
                    # Skip figures with empty or generic captions
                    if not caption or re.match(r"^Figure on p\.\d+$", caption):
                        continue
                    chunk_text += f"\n\n![{caption}]({fig.image_url})\n*{caption}*"

        parts.append(chunk_text)
    return "\n\n".join(parts)