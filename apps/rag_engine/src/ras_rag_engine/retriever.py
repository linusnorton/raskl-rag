"""RAG retrieval: embed query → pgvector cosine search."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from .config import RAGConfig
from .providers import get_embed_provider

log = logging.getLogger(__name__)


@dataclass
class RetrievedFigure:
    figure_id: str
    doc_id: str
    page_num: int
    caption: str
    image_url: str
    thumb_url: str
    source_filename: str
    asset_path: str = ""


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    start_page: int
    end_page: int
    section_heading: str | None
    source_filename: str
    title: str | None
    author: str | None
    year: int | None
    publication: str | None = None
    document_type: str | None = None
    page_offset: int = 0
    editor: str | None = None
    abstract: str | None = None
    description: str | None = None


def embed_query(query: str, config: RAGConfig) -> list[float]:
    """Embed a single query string via the configured provider."""
    provider = get_embed_provider(config)
    embeddings = provider.embed([query])
    return embeddings[0]


_FTS_COUNT_SQL = """\
SELECT count(*) FROM chunks c
WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', %(query)s)
"""

RETRIEVE_SQL = """\
WITH vector_results AS (
    SELECT chunk_id, doc_id, text, start_page, end_page, section_heading,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS vrank
    FROM chunks
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT 50
),
text_results AS (
    SELECT c.chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(
               setweight(to_tsvector('english', c.text), 'B')
               || setweight(to_tsvector('english',
                   coalesce(d.title, '') || ' ' || coalesce(d.author, '') || ' '
                   || coalesce(d.abstract, '') || ' ' || coalesce(d.description, '') || ' '
                   || coalesce(array_to_string(d.keywords, ' '), '')), 'A'),
               %(tsquery)s::tsquery) DESC) AS trank
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE to_tsvector('english', c.text) @@ %(tsquery)s::tsquery
    LIMIT 50
),
fused AS (
    SELECT v.chunk_id, v.doc_id, v.text, v.start_page, v.end_page, v.section_heading,
           COALESCE(1.0 / (60 + v.vrank), 0) + COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
    FROM vector_results v
    LEFT JOIN text_results t ON v.chunk_id = t.chunk_id
    UNION
    SELECT c.chunk_id, c.doc_id, c.text, c.start_page, c.end_page, c.section_heading,
           COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
    FROM text_results t
    JOIN chunks c ON t.chunk_id = c.chunk_id
    WHERE t.chunk_id NOT IN (SELECT chunk_id FROM vector_results)
)
SELECT f.chunk_id, f.doc_id, f.text, f.start_page, f.end_page, f.section_heading,
       d.source_filename, d.title, d.author, d.year, d.page_offset, d.publication, d.document_type,
       f.rrf_score AS score, d.editor, d.abstract, d.description
FROM fused f
JOIN documents d ON f.doc_id = d.doc_id
ORDER BY f.rrf_score DESC
LIMIT %(top_k)s
"""

# Minimum AND-based FTS matches before falling back to OR
_FTS_AND_MIN_MATCHES = 10


def _build_tsquery(query: str, cur: psycopg.Cursor) -> str:
    """Build a tsquery string, trying AND first and falling back to OR if too few matches."""
    # Get the AND-based tsquery (PostgreSQL default for plainto_tsquery)
    cur.execute("SELECT plainto_tsquery('english', %(q)s)::text", {"q": query})
    and_query = cur.fetchone()[0]

    if not and_query:
        return and_query

    # Count AND matches
    cur.execute(_FTS_COUNT_SQL, {"query": query})
    and_count = cur.fetchone()[0]

    if and_count >= _FTS_AND_MIN_MATCHES:
        log.info("FTS AND query matched %d chunks, using AND", and_count)
        return and_query

    # Fall back to OR
    or_query = and_query.replace("&", "|")
    log.info("FTS AND query matched only %d chunks (< %d), falling back to OR", and_count, _FTS_AND_MIN_MATCHES)
    return or_query


def retrieve(query: str, config: RAGConfig, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed the query and retrieve the most similar chunks via hybrid search (vector + full-text RRF)."""
    top_k = top_k or config.retrieval_top_k
    fetch_k = config.rerank_candidates if config.rerank_enabled else top_k
    vec = embed_query(query, config)
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            tsquery = _build_tsquery(query, cur)
            cur.execute(RETRIEVE_SQL, {"vec": vec_str, "tsquery": tsquery, "top_k": fetch_k})
            rows = cur.fetchall()

    chunks = []
    for row in rows:
        chunks.append(RetrievedChunk(
            chunk_id=row[0],
            doc_id=row[1],
            text=row[2],
            start_page=row[3],
            end_page=row[4],
            section_heading=row[5],
            source_filename=row[6],
            title=row[7],
            author=row[8],
            year=row[9],
            page_offset=row[10],
            publication=row[11],
            document_type=row[12],
            score=row[13],
            editor=row[14],
            abstract=row[15],
            description=row[16],
        ))

    if config.rerank_enabled and chunks:
        from .reranker import rerank
        chunks = rerank(query, chunks, config, top_k=top_k)

    return chunks


def retrieve_contextual_figures(chunks: list[RetrievedChunk], config: RAGConfig) -> list[RetrievedFigure]:
    """Retrieve figures on the same pages as the given chunks.

    In local mode (no s3_bucket), filters out figures whose asset files don't exist on disk.
    """
    doc_id_pages: set[tuple[str, int]] = set()
    for c in chunks:
        for p in range(c.start_page, c.end_page + 1):
            doc_id_pages.add((c.doc_id, p))

    if not doc_id_pages:
        return []

    base = config.api_base_url.rstrip("/")
    pairs = list(doc_id_pages)
    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.figure_id, f.doc_id, f.page_num, f.caption, f.asset_path, f.thumb_path,
                       d.source_filename
                FROM figures f
                JOIN documents d ON f.doc_id = d.doc_id
                WHERE (f.doc_id, f.page_num) IN (SELECT unnest(%(doc_ids)s::text[]), unnest(%(pages)s::int[]))
                """,
                {"doc_ids": [p[0] for p in pairs], "pages": [p[1] for p in pairs]},
            )
            rows = cur.fetchall()

    from pathlib import Path

    is_local = not config.s3_bucket
    data_dir = Path(config.data_dir)

    figures = []
    for r in rows:
        figure_id, doc_id, page_num, caption, asset_path, thumb_path, source_filename = r

        # In local mode, skip figures whose asset file doesn't exist
        if is_local and asset_path:
            local_path = data_dir / doc_id / asset_path
            if not local_path.is_file():
                log.debug("Skipping figure %s: asset not found at %s", figure_id, local_path)
                continue

        figures.append(RetrievedFigure(
            figure_id=figure_id,
            doc_id=doc_id,
            page_num=page_num,
            caption=caption,
            image_url=f"{base}/v1/images/{figure_id}",
            thumb_url=f"{base}/v1/images/{figure_id}?thumb=true",
            source_filename=source_filename,
            asset_path=asset_path or "",
        ))

    return figures


_FIGURE_SEARCH_SQL = """\
WITH vector_results AS (
    SELECT figure_id, doc_id, page_num, caption, asset_path, thumb_path,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS vrank
    FROM figures
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT 30
),
text_results AS (
    SELECT figure_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', caption), plainto_tsquery('english', %(query)s)) DESC) AS trank
    FROM figures
    WHERE to_tsvector('english', caption) @@ plainto_tsquery('english', %(query)s)
    LIMIT 30
),
fused AS (
    SELECT v.figure_id, v.doc_id, v.page_num, v.caption,
           COALESCE(1.0 / (60 + v.vrank), 0) + COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
    FROM vector_results v
    LEFT JOIN text_results t ON v.figure_id = t.figure_id
    UNION
    SELECT f.figure_id, f.doc_id, f.page_num, f.caption,
           COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
    FROM text_results t
    JOIN figures f ON t.figure_id = f.figure_id
    WHERE t.figure_id NOT IN (SELECT figure_id FROM vector_results)
)
SELECT fused.figure_id, fused.doc_id, fused.page_num, fused.caption,
       d.source_filename, fused.rrf_score
FROM fused
JOIN documents d ON fused.doc_id = d.doc_id
ORDER BY fused.rrf_score DESC
LIMIT %(top_k)s
"""


def retrieve_figures(query: str, config: RAGConfig, top_k: int = 5) -> list[RetrievedFigure]:
    """Search for figures using hybrid search (vector + FTS on captions)."""
    vec = embed_query(query, config)
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    base = config.api_base_url.rstrip("/")
    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_FIGURE_SEARCH_SQL, {"vec": vec_str, "query": query, "top_k": top_k})
            rows = cur.fetchall()

    return [
        RetrievedFigure(
            figure_id=row[0],
            doc_id=row[1],
            page_num=row[2],
            caption=row[3],
            image_url=f"{base}/v1/images/{row[0]}",
            thumb_url=f"{base}/v1/images/{row[0]}?thumb=true",
            source_filename=row[4],
        )
        for row in rows
    ]
