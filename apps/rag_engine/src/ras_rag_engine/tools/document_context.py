from __future__ import annotations
import psycopg
from ..config import RAGConfig
from ..retriever import RetrievedChunk

DEFINITION = {
    "type": "function",
    "function": {
        "name": "document_context",
        "description": (
            "Retrieve additional surrounding text from a specific document. "
            "Use this when a search result [N] is too brief and you need to "
            "read the preceding or following pages to understand the full context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "The unique ID of the document (found in the source metadata)."
                },
                "page_num": {
                    "type": "integer",
                    "description": "The page number to focus on."
                },
                "page_window": {
                    "type": "integer",
                    "description": "How many pages to retrieve before and after (default is 1).",
                    "default": 1
                }
            },
            "required": ["doc_id", "page_num"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Fetch sequential chunks from a document by page range."""
    doc_id = args["doc_id"]
    page_num = args["page_num"]
    window = args.get("page_window", 1)

    # Define the range to fetch
    min_page = max(1, page_num - window)
    max_page = page_num + window

    # Query the chunks table directly. We sort by start_page and chunk_id 
    # to ensure the text flows correctly for the reader.
    query = """
        SELECT c.chunk_id, c.doc_id, c.text, c.start_page, c.end_page, 
               c.section_heading, d.source_filename, d.title, d.author, d.year
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE c.doc_id = %(doc_id)s 
          AND c.start_page >= %(min_page)s 
          AND c.start_page <= %(max_page)s
        ORDER BY c.start_page ASC, c.chunk_id ASC
        LIMIT 10
    """

    chunks = []
    with psycopg.connect(config.database_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, {
                "doc_id": doc_id, 
                "min_page": min_page, 
                "max_page": max_page
            })
            for row in cur.fetchall():
                chunks.append(RetrievedChunk(
                    chunk_id=row[0], doc_id=row[1], text=row[2], score=1.0,
                    start_page=row[3], end_page=row[4], section_heading=row[5],
                    source_filename=row[6], title=row[7], author=row[8], year=row[9]
                ))

    if not chunks:
        return f"No additional context found for document {doc_id} on pages {min_page}-{max_page}.", []

    lines = [f"Retrieved {len(chunks)} contextual chunks from {chunks[0].source_filename}:"]
    for i, c in enumerate(chunks, start=start_index):
        lines.append(f"[{i}] (Page {c.start_page}):\n{c.text}\n---")

    return "\n\n".join(lines), chunks