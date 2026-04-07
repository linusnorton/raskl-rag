from ..config import RAGConfig
from .search_basic import execute as execute_basic_search

DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_by_attribute",
        "description": (
            "High-precision search using document attributes. "
            "Use this when the query includes a specific year, "
            "document type (e.g., 'obituary'), or publication."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term."},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "document_type": {"type": "string"},
                "publication": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}

def execute(args, config: RAGConfig, start_index: int = 1):
    # Reuse your existing robust search logic from search_basic
    return execute_basic_search(args, config, start_index=start_index)