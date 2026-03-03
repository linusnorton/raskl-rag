"""Tool definitions and executors for the RAG agent."""

from __future__ import annotations

import json

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from duckduckgo_search import DDGS

from .config import ChatConfig
from .retriever import RetrievedChunk, retrieve

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
                }
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
TOOL_DEFINITIONS = [_SEARCH_TOOL, _WEB_SEARCH_TOOL]


def get_tool_definitions(config: ChatConfig) -> list[dict]:
    """Return tool definitions based on config (web_search conditional)."""
    tools = [_SEARCH_TOOL]
    if config.web_search_enabled:
        tools.append(_WEB_SEARCH_TOOL)
    return tools


def format_chunks_for_context(chunks: list[RetrievedChunk], start_index: int = 1) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM."""
    if not chunks:
        return "(No relevant passages found.)"

    parts = []
    for i, c in enumerate(chunks, start=start_index):
        source = c.title or c.source_filename
        display_start = c.start_page
        display_end = c.end_page
        pages = f"p.{display_start}" if display_start == display_end else f"pp.{display_start}-{display_end}"
        heading = f" — {c.section_heading}" if c.section_heading else ""
        author_year = ""
        if c.author or c.year:
            parts_ay = []
            if c.author:
                parts_ay.append(c.author)
            if c.year:
                parts_ay.append(str(c.year))
            author_year = f" ({', '.join(parts_ay)})"
        header = f"[{i}] {source}{author_year}, {pages}{heading}"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def _execute_search_documents(args: dict, config: ChatConfig) -> tuple[str, list[RetrievedChunk]]:
    """Execute a document search and return formatted results + raw chunks."""
    query = args["query"]
    chunks = retrieve(query, config)
    text = format_chunks_for_context(chunks)
    return text, chunks


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


def execute_tool_call(name: str, args_json: str, config: ChatConfig) -> tuple[str, list[RetrievedChunk]]:
    """Dispatch a tool call by name. Returns (result_text, retrieved_chunks)."""
    args = json.loads(args_json) if isinstance(args_json, str) else args_json

    if name == "search_documents":
        return _execute_search_documents(args, config)
    elif name == "web_search":
        return _execute_web_search(args)
    else:
        return f"Unknown tool: {name}", []
