from __future__ import annotations
from ..config import RAGConfig
from ..retriever import RetrievedChunk, retrieve

DEFINITION = {
    "type": "function",
    "function": {
        "name": "mbras_index",
        "description": (
            "Consult the master MBRAS publication index to find article titles, "
            "canonical author names, or subject categories. Use this as a first step "
            "to find out what an author has written or what articles exist on a topic."
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
    # This tool is efficient because it targets a single source.
    chunks = retrieve(
        args["query"],
        config,
        source_filename="Index_1878-2025-Je.pdf"
    )

    if not chunks:
        return "No index entries found for that query.", []

    lines = ["Found the following entries in the MBRAS Master Index:"]
    for i, c in enumerate(chunks, start=start_index):
        lines.append(f"[{i}] {c.text}\n---")
    
    return "\n".join(lines), chunks