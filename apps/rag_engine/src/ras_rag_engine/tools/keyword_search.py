from __future__ import annotations
from ..config import RAGConfig
from ..retriever import RetrievedChunk, retrieve

DEFINITION = {
    "type": "function",
    "function": {
        "name": "keyword_search",
        "description": (
            "Perform a high-precision keyword search for specific names, rare terms, or unique spellings. "
            "Use this when a general topic search fails to find a specific person or event, "
            "or when the exact phrasing is critical (e.g., searching for 'Ralph Fitch')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The exact name, term, or phrase to find in the documents.",
                    "maxLength": 100
                },
                "document_type": {"type": "string", "description": "Optional filter (e.g., 'primary_source')."},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"}
            },
            "required": ["keyword"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Execute a precision search using Postgres Full-Text Search (FTS)."""
    keyword = args["keyword"]
    doc_type = args.get("document_type")
    year_from = args.get("year_from")
    year_to = args.get("year_to")

    # We use the existing 'retrieve' function but pass the keyword 
    # as the query. In your current retriever.py, this triggers the 
    # hybrid search logic which includes FTS weighting.
    chunks = retrieve(
        keyword,
        config,
        document_type=doc_type,
        year_from=year_from,
        year_to=year_to
    )

    if not chunks:
        return f"No exact matches found for keyword: '{keyword}'.", []

    lines = [f"Found {len(chunks)} exact matches for '{keyword}':"]
    for i, c in enumerate(chunks, start=start_index):
        header = f"[{i}] SOURCE: {c.source_filename}"
        if c.title:
            header += f" | TITLE: {c.title}"
        if c.year:
            header += f" ({c.year})"
        lines.append(f"{header}\n{c.text}\n---")

    return "\n\n".join(lines), chunks