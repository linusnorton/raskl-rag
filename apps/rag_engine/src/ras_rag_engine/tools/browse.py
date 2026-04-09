from __future__ import annotations
import psycopg
from ..config import RAGConfig
from ..retriever import RetrievedChunk
from .utils import resolve_publication_pattern, resolve_author_pattern, get_type_label

_BROWSE_NO_CITE_NOTE = "\n\nNOTE: The list above are catalogue entries. Do NOT use [N] citations for them."

DEFINITION = {
    "type": "function",
    "function": {
        "name": "browse_corpus",
        "description": "List documents in the collection or get metadata summaries without retrieving full text.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "summarize"], "default": "list"},
                "publication": {"type": "string"},
                "author": {"type": "string"},
                "year": {"type": "integer"}
            },
            "required": ["action"]
        }
    }
}

def execute(args: dict, config: RAGConfig, start_index: int = 1) -> tuple[str, list[RetrievedChunk]]:
    """Browse document metadata directly from the database."""
    action = args.get("action", "list")
    pub = resolve_publication_pattern(args.get("publication"))
    author = args.get("author")
    year = args.get("year")

    query = "SELECT title, author, year, publication, document_type FROM documents WHERE 1=1"
    params = {}
    
    if pub:
        query += " AND publication ILIKE %(pub)s"
        params["pub"] = pub
    if author:
        name_parts = resolve_author_pattern(author)
        for i, part in enumerate(name_parts):
            key = f"auth_{i}"
            query += f" AND author ILIKE %({key})s"
            params[key] = f"%{part}%"
    if year:
        query += " AND year = %(year)s"
        params["year"] = year
        
    query += " LIMIT 25"

    with psycopg.connect(config.database_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
    if not rows:
        return "No documents match those filters.", []

    lines = [f"Found {len(rows)} documents:"]
    for i, row in enumerate(rows, 1):
        type_label = get_type_label(row[4]).strip(" []")
        lines.append(f"{i}. {row[0]} by {row[1]} ({row[2]}) [{type_label}]")
        
    return "\n".join(lines) + _BROWSE_NO_CITE_NOTE, []