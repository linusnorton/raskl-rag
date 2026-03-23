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

_BROWSE_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "browse_corpus",
        "description": (
            "Browse and explore the document collection metadata. Use this when the user asks "
            "about what documents are available, wants to list contents of a specific volume or "
            "issue, count documents by publication or year, or get an overview of the corpus. "
            "Do NOT use this for content-based questions — use search_documents for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "count", "overview"],
                    "description": (
                        "Action to perform: "
                        "'list' — list documents matching filters (returns title, author, year, volume, issue, pages, type). "
                        "'count' — count documents grouped by a field (publication, year, document_type, volume, author). "
                        "'overview' — summary statistics of the entire corpus."
                    ),
                },
                "publication": {
                    "type": "string",
                    "description": "Filter by publication name (e.g. 'JMBRAS', 'JSBRAS'). Matches loosely.",
                },
                "volume": {
                    "type": "string",
                    "description": "Filter by volume number (e.g. '87').",
                },
                "issue": {
                    "type": "string",
                    "description": "Filter by issue/part number (e.g. '1', '2').",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Lower bound (inclusive) for publication year.",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Upper bound (inclusive) for publication year.",
                },
                "document_type": {
                    "type": "string",
                    "description": (
                        "Filter by document type: journal_article, front_matter, obituary, editors_note, "
                        "annual_report, agm_minutes, biographical_notes, secondary_source, "
                        "primary_source, mbras_monograph, mbras_reprint, index, illustration."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author name (partial match, e.g. 'Swettenham').",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["publication", "year", "document_type", "volume", "author"],
                    "description": "For 'count' action: field to group results by. Defaults to 'publication'.",
                },
            },
            "required": ["action"],
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
TOOL_DEFINITIONS = [_SEARCH_TOOL, _FIND_IMAGES_TOOL, _BROWSE_CORPUS_TOOL, _WEB_SEARCH_TOOL]


def get_tool_definitions(config: RAGConfig) -> list[dict]:
    """Return tool definitions based on config (web_search conditional)."""
    tools = [_SEARCH_TOOL, _FIND_IMAGES_TOOL, _BROWSE_CORPUS_TOOL]
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


# Publication name aliases for ILIKE matching (the publication field is inconsistent)
_PUBLICATION_PATTERNS = {
    "JMBRAS": "%Royal Asiatic%",
    "JMRAS": "%Royal Asiatic%",
    "JSBRAS": "%Straits Branch%",
    "MBRAS": "%Royal Asiatic%",
}


def _resolve_publication_pattern(value: str | None) -> str | None:
    """Convert a publication filter to an ILIKE pattern."""
    if not value:
        return None
    return _PUBLICATION_PATTERNS.get(value.upper(), f"%{value}%")


def _build_where_clause(args: dict) -> tuple[str, dict]:
    """Build a WHERE clause and params from browse_corpus filter args."""
    conditions = []
    params: dict = {}
    pub_pattern = _resolve_publication_pattern(args.get("publication"))
    if pub_pattern:
        conditions.append("publication ILIKE %(pub)s")
        params["pub"] = pub_pattern
    if args.get("volume"):
        conditions.append("volume = %(volume)s")
        params["volume"] = args["volume"]
    if args.get("issue"):
        conditions.append("issue = %(issue)s")
        params["issue"] = args["issue"]
    if args.get("year_from"):
        conditions.append("year >= %(year_from)s")
        params["year_from"] = args["year_from"]
    if args.get("year_to"):
        conditions.append("year <= %(year_to)s")
        params["year_to"] = args["year_to"]
    if args.get("document_type"):
        conditions.append("document_type = %(doc_type)s")
        params["doc_type"] = args["document_type"]
    if args.get("author"):
        conditions.append("author ILIKE %(author)s")
        params["author"] = f"%{args['author']}%"
    where = " AND ".join(conditions) if conditions else "TRUE"
    return where, params


_BROWSE_NO_CITE_NOTE = (
    "\n\n(These results are from a metadata catalogue query. "
    "Do NOT use [N] citation markers for this information.)"
)


def _execute_browse_corpus(args: dict, config: RAGConfig) -> tuple[str, list[RetrievedChunk]]:
    """Execute a corpus browse/introspection query."""
    import psycopg

    action = args.get("action", "list")
    where, params = _build_where_clause(args)

    with psycopg.connect(config.dsn) as conn:
        if action == "overview":
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS total,
                       COUNT(DISTINCT publication) AS pubs,
                       MIN(year) AS earliest,
                       MAX(year) AS latest,
                       COUNT(DISTINCT volume) FILTER (WHERE publication ILIKE '%%Royal Asiatic%%') AS jmbras_vols,
                       COUNT(*) FILTER (WHERE publication ILIKE '%%Royal Asiatic%%') AS jmbras_docs,
                       COUNT(*) FILTER (WHERE publication ILIKE '%%Straits Branch%%') AS jsbras_docs
                FROM documents WHERE {where}
                """,
                params,
            ).fetchone()
            total, pubs, earliest, latest, jmbras_vols, jmbras_docs, jsbras_docs = row
            lines = [
                f"Corpus overview:",
                f"- Total documents: {total}",
                f"- Distinct publications: {pubs}",
                f"- Year range: {earliest or '?'}–{latest or '?'}",
                f"- JMBRAS/JMRAS/JSBRAS journals: {jmbras_docs + jsbras_docs} documents across {jmbras_vols} volumes",
            ]
            # Add document type breakdown
            rows = conn.execute(
                f"""
                SELECT document_type, COUNT(*) AS cnt
                FROM documents WHERE {where}
                GROUP BY document_type ORDER BY cnt DESC
                """,
                params,
            ).fetchall()
            if rows:
                lines.append("- By type:")
                for dtype, cnt in rows:
                    label = _TYPE_LABELS.get(dtype, f" [{dtype}]").strip(" []") if dtype else "Unknown"
                    lines.append(f"  {label}: {cnt}")
            return "\n".join(lines) + _BROWSE_NO_CITE_NOTE, []

        elif action == "count":
            group_by = args.get("group_by", "publication")
            valid_groups = {"publication", "year", "document_type", "volume", "author"}
            if group_by not in valid_groups:
                group_by = "publication"
            order = "cnt DESC" if group_by != "year" else f"{group_by}"
            rows = conn.execute(
                f"""
                SELECT {group_by}, COUNT(*) AS cnt
                FROM documents WHERE {where}
                GROUP BY {group_by} ORDER BY {order}
                LIMIT 100
                """,
                params,
            ).fetchall()
            if not rows:
                return "No documents found matching those filters.", []
            lines = [f"Document count by {group_by}:"]
            for val, cnt in rows:
                display = val or "Unknown"
                lines.append(f"- {display}: {cnt}")
            return "\n".join(lines) + _BROWSE_NO_CITE_NOTE, []

        else:  # action == "list"
            rows = conn.execute(
                f"""
                SELECT title, author, year, volume, issue, document_type, page_range_label, source_filename
                FROM documents WHERE {where}
                ORDER BY year NULLS LAST, volume NULLS LAST, issue NULLS LAST, title NULLS LAST
                LIMIT 100
                """,
                params,
            ).fetchall()
            if not rows:
                return "No documents found matching those filters.", []
            lines = [f"Found {len(rows)} document(s):"]
            for i, (title, author, year, volume, issue, doc_type, pages, filename) in enumerate(rows, 1):
                display_title = title or filename
                type_label = _TYPE_LABELS.get(doc_type, "").strip(" []")
                parts = []
                if author:
                    parts.append(author)
                if year:
                    parts.append(f"({year})")
                parts.append(f'"{display_title}"')
                if volume and issue:
                    parts.append(f"Vol.{volume} Part {issue}")
                elif volume:
                    parts.append(f"Vol.{volume}")
                if pages:
                    parts.append(f"pp.{pages}")
                if type_label:
                    parts.append(f"[{type_label}]")
                lines.append(f"{i}. {', '.join(parts)}")
            return "\n".join(lines) + _BROWSE_NO_CITE_NOTE, []


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
    elif name == "browse_corpus":
        return _execute_browse_corpus(args, config)
    elif name == "web_search":
        return _execute_web_search(args)
    else:
        return f"Unknown tool: {name}", []
