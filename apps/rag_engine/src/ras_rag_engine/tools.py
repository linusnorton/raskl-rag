"""Tool definitions and executors for the RAG agent."""

from __future__ import annotations

import json
import re
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from duckduckgo_search import DDGS

from .config import RAGConfig
from .retriever import RetrievedChunk, RetrievedFigure, retrieve, retrieve_figures

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the indexed JMBRAS/Swettenham document collection for passages relevant to a query. "
            "Use this when you need more context on a specific topic, person, place, or event from the documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific search query describing what information you need from the documents.",
                },
                "document_type": {
                    "type": "string",
                    "description": (
                        "Optional filter to restrict results to a specific document type. "
                        "Valid values: journal_article, front_matter, obituary, editors_note, "
                        "annual_report, agm_minutes, biographical_notes, secondary_source, "
                        "primary_source, mbras_monograph, mbras_reprint, index, illustration."
                    ),
                },
                "year_from": {
                    "type": "integer",
                    "description": "Optional lower bound (inclusive) for publication year.",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Optional upper bound (inclusive) for publication year.",
                },
                "language": {
                    "type": "string",
                    "description": "Optional ISO language code filter: en, ms, zh, ar.",
                },
                "publication": {
                    "type": "string",
                    "description": "Optional exact-match filter for publication name (e.g. JMBRAS, JSBRAS).",
                },
            },
            "required": ["query"],
        },
    },
}

_FIND_IMAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "find_images",
        "description": (
            "Search for images, figures, plates, maps, and photographs in the JMBRAS/Swettenham collection. "
            "Use when the user asks to see or find a specific image, figure, plate, map, or photograph."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A search query describing the image or figure you are looking for.",
                },
                "document_type": {
                    "type": "string",
                    "description": "Optional filter to restrict results to a specific document type.",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Optional lower bound (inclusive) for publication year.",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Optional upper bound (inclusive) for publication year.",
                },
            },
            "required": ["query"],
        },
    },
}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for general knowledge, historical context, or information not found in the document collection. "
            "Use this as a fallback when document search returns nothing relevant, or for background context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                }
            },
            "required": ["query"],
        },
    },
}

# Keep backward-compatible module-level constant (includes all tools)
TOOL_DEFINITIONS = [_SEARCH_TOOL, _FIND_IMAGES_TOOL, _WEB_SEARCH_TOOL]


def get_tool_definitions(config: RAGConfig) -> list[dict]:
    """Return tool definitions based on config (web_search conditional)."""
    tools = [_SEARCH_TOOL, _FIND_IMAGES_TOOL]
    if config.web_search_enabled:
        tools.append(_WEB_SEARCH_TOOL)
    return tools


def _parse_filename_metadata(filename: str) -> tuple[str, str]:
    """Extract author and year from source_filename like 'Abdullah (2011) JMBRAS 84(1), 1-22.pdf'."""
    import re

    m = re.match(r"^(.+?)\s*\((\d{4})\)", filename)
    if m:
        return m.group(1).strip(), m.group(2)
    return "", ""


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
        type_label = _TYPE_LABELS.get(c.document_type, "")
        # Use DB metadata, falling back to parsing the source filename
        fn_author, fn_year = _parse_filename_metadata(c.source_filename)
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


def _execute_search_documents(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Execute a document search and return formatted results + raw chunks."""
    query = args["query"]
    document_type = args.get("document_type")
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    language = args.get("language")
    publication = args.get("publication")
    chunks = retrieve(
        query, config, document_type=document_type,
        year_from=year_from, year_to=year_to, language=language, publication=publication,
    )
    text = format_chunks_for_context(chunks, start_index=start_index)
    return text, chunks


def _execute_find_images(args: dict, config: RAGConfig) -> tuple[str, list[RetrievedChunk]]:
    """Execute an image search and return formatted results with markdown image syntax."""
    query = args["query"]
    document_type = args.get("document_type")
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    figures = retrieve_figures(
        query, config, top_k=5,
        document_type=document_type, year_from=year_from, year_to=year_to,
    )

    if not figures:
        return "No images found matching that query.", []

    parts = []
    for fig in figures:
        caption = fig.caption or "Untitled figure"
        parts.append(f"**{caption}** (p.{fig.page_num}, {fig.source_filename})\n![{caption}]({fig.image_url})")
    return "\n\n".join(parts), []


def _execute_web_search(args: dict) -> tuple[str, list[RetrievedChunk]]:
    """Execute a web search and return formatted results."""
    query = args["query"]
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Web search failed: {e}", []

    if not results:
        return "No web results found.", []

    parts = []
    for r in results:
        parts.append(f"**{r['title']}**\n{r['body']}\n{r['href']}")
    return "\n\n".join(parts), []


def execute_tool_call(name: str, args_json: str, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Dispatch a tool call by name. Returns (result_text, retrieved_chunks)."""
    args = json.loads(args_json) if isinstance(args_json, str) else args_json

    if name == "search_documents":
        return _execute_search_documents(args, config, start_index=start_index)
    elif name == "find_images":
        return _execute_find_images(args, config)
    elif name == "web_search":
        return _execute_web_search(args)
    else:
        return f"Unknown tool: {name}", []
