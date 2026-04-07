# apps/rag_engine/src/ras_rag_engine/tools/figures.py
from __future__ import annotations
from ..config import RAGConfig
from ..retriever import RetrievedChunk, retrieve_figures

DEFINITION = {
    "type": "function",
    "function": {
        "name": "find_images",
        "description": "Search for relevant figures, illustrations, or photographs in the MBRAS collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of the image needed."},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"}
            },
            "required": ["query"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Search for figures using hybrid vector + FTS search on captions."""
    query = args["query"]
    year_from = args.get("year_from")
    year_to = args.get("year_to")

    figures = retrieve_figures(
        query,
        config,
        year_from=year_from,
        year_to=year_to
    )

    if not figures:
        return "No relevant images found.", []

    lines = []
    # Convert figures to chunks for the agent context 
    as_chunks = []
    for i, f in enumerate(figures, start=start_index):
        lines.append(f"[{i}] FIGURE: {f.caption} (Source: {f.source_filename}, Page {f.page_num})")
    
    return "The following figures were found:\n" + "\n".join(lines), as_chunks