from __future__ import annotations
from ..config import RAGConfig
from ..retriever import RetrievedChunk, retrieve
from .utils import resolve_author_pattern

DEFINITION = {
    "type": "function",
    "function": {
        "name": "mbras_index",
        "description": (
            "Search the MBRAS Master Index (1878-2025) for specific authors, subjects, or keywords. DO NOT use this for general journal listings or metadata queries; use browse_corpus for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The author name or subject to find."}
            },
            "required": ["query"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    query = args["query"]
    parts = resolve_author_pattern(query)
    if parts:
        query = " ".join(parts)
    chunks = retrieve(
        query,
        config,
        source_filename="Index_1878-2025-Je.pdf"
    )

    if not chunks:
        return "No index entries found for that query.", []

    lines = ["Found the following entries in the MBRAS Master Index:"]
    for i, c in enumerate(chunks, start=start_index):
        lines.append(f"[{i}] {c.text}\n---")
    
    return "\n".join(lines), chunks