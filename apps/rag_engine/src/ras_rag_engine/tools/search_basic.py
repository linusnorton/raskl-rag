from __future__ import annotations
from ..config import RAGConfig
from ..retriever import RetrievedChunk, retrieve
from .utils import resolve_publication_pattern

DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the indexed MBRAS document collection for passages relevant to a query. "
            "Use this when you need more context on a specific topic, person, place, or event."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific search query describing what information you need.",
                    "maxLength": 255
                },
                "document_type": {
                    "type": "string",
                    "description": "Optional filter (e.g., 'journal_article', 'obituary', 'primary_source')."
                },
                "year_from": {"type": "integer", "description": "Optional start year filter."},
                "year_to": {"type": "integer", "description": "Optional end year filter."},
                "publication": {"type": "string", "description": "Optional publication filter (e.g., 'JMBRAS')."}
            },
            "required": ["query"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Execute the standard vector-based document search."""
    query = args["query"]
    doc_type = args.get("document_type")
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    publication = resolve_publication_pattern(args.get("publication"))

    chunks = retrieve(
        query,
        config,
        document_type=doc_type,
        year_from=year_from,
        year_to=year_to,
        publication=publication
    )

    if not chunks:
        return "No relevant documents found for that query.", []

    # Format findings with citation markers for the agent 
    lines = []
    for i, c in enumerate(chunks, start=start_index):
        header = f"[{i}] SOURCE: {c.source_filename}"
        if c.title:
            header += f" | TITLE: {c.title}"
        if c.author:
            header += f" | AUTHOR: {c.author}"
        if c.year:
            header += f" ({c.year})"
        
        lines.append(f"{header}\n{c.text}\n---")

    return "\n\n".join(lines), chunks