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
    SELECT c.chunk_id, c.doc_id, c.text, c.start_page, c.end_page, c.section_heading,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(vec)s::vector) AS vrank
    FROM chunks c
    JOIN documents d_v ON c.doc_id = d_v.doc_id
    WHERE (%(doc_type)s::text IS NULL OR d_v.document_type = %(doc_type)s)
      AND (%(year_from)s::int IS NULL OR d_v.year >= %(year_from)s)
      AND (%(year_to)s::int IS NULL OR d_v.year <= %(year_to)s)
      AND (%(language)s::text IS NULL OR d_v.language = %(language)s)
      AND (%(publication)s::text IS NULL OR d_v.publication = %(publication)s)
      AND (%(source_filename)s::text IS NULL OR d_v.source_filename = %(source_filename)s)
    ORDER BY c.embedding <=> %(vec)s::vector
    LIMIT 100
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
      AND (%(doc_type)s::text IS NULL OR d.document_type = %(doc_type)s)
      AND (%(year_from)s::int IS NULL OR d.year >= %(year_from)s)
      AND (%(year_to)s::int IS NULL OR d.year <= %(year_to)s)
      AND (%(language)s::text IS NULL OR d.language = %(language)s)
      AND (%(publication)s::text IS NULL OR d.publication = %(publication)s)
      AND (%(source_filename)s::text IS NULL OR d.source_filename = %(source_filename)s)
    LIMIT 100
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


def _apply_doc_cap(chunks: list[RetrievedChunk], max_per_doc: int, target_count: int) -> list[RetrievedChunk]:
    """Cap chunks per document to ensure source diversity. Chunks must be pre-sorted by score (desc)."""
    if max_per_doc <= 0:
        return chunks[:target_count]

    doc_counts: dict[str, int] = {}
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        count = doc_counts.get(chunk.doc_id, 0)
        if count >= max_per_doc:
            continue
        doc_counts[chunk.doc_id] = count + 1
        result.append(chunk)
        if len(result) >= target_count:
            break

    doc_dist = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info(
        "Diversity cap applied: %d candidates from %d docs (max %d/doc), top-5: %s",
        len(result), len(doc_counts), max_per_doc,
        ", ".join(f"{did}={n}" for did, n in doc_dist),
    )
    return result


def retrieve(
    query: str,
    config: RAGConfig,
    top_k: int | None = None,
    document_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
    publication: str | None = None,
    source_filename: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and retrieve the most similar chunks via hybrid search (vector + full-text RRF)."""
    query = query[:2000] # Character limit for retrieval queries
    top_k = top_k or config.retrieval_top_k
    fetch_k = config.rerank_candidates if config.rerank_enabled else top_k
    # When diversity cap is active, fetch extra from SQL so the cap has enough to work with
    sql_limit = fetch_k * 3 if config.diversity_max_per_doc > 0 else fetch_k
    vec = embed_query(query, config)
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            tsquery = _build_tsquery(query, cur)
            cur.execute(RETRIEVE_SQL, {
                "vec": vec_str, "tsquery": tsquery, "top_k": sql_limit, "doc_type": document_type, "source_filename": source_filename,
                "year_from": year_from, "year_to": year_to, "language": language, "publication": publication, 
            })
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

    if config.diversity_max_per_doc > 0 and chunks:
        chunks = _apply_doc_cap(chunks, config.diversity_max_per_doc, fetch_k)

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
    SELECT f.figure_id, f.doc_id, f.page_num, f.caption, f.asset_path, f.thumb_path,
           ROW_NUMBER() OVER (ORDER BY f.embedding <=> %(vec)s::vector) AS vrank
    FROM figures f
    JOIN documents d_v ON f.doc_id = d_v.doc_id
    WHERE f.embedding IS NOT NULL
      AND (%(doc_type)s::text IS NULL OR d_v.document_type = %(doc_type)s)
      AND (%(year_from)s::int IS NULL OR d_v.year >= %(year_from)s)
      AND (%(year_to)s::int IS NULL OR d_v.year <= %(year_to)s)
    ORDER BY f.embedding <=> %(vec)s::vector
    LIMIT 30
),
text_results AS (
    SELECT f.figure_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(
               to_tsvector('english', f.caption),
               plainto_tsquery('english', %(query)s)) DESC) AS trank
    FROM figures f
    JOIN documents d_t ON f.doc_id = d_t.doc_id
    WHERE to_tsvector('english', f.caption) @@ plainto_tsquery('english', %(query)s)
      AND (%(doc_type)s::text IS NULL OR d_t.document_type = %(doc_type)s)
      AND (%(year_from)s::int IS NULL OR d_t.year >= %(year_from)s)
      AND (%(year_to)s::int IS NULL OR d_t.year <= %(year_to)s)
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


def retrieve_figures(
    query: str,
    config: RAGConfig,
    top_k: int = 5,
    document_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[RetrievedFigure]:
    """Search for figures using hybrid search (vector + FTS on captions)."""
    vec = embed_query(query, config)
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    base = config.api_base_url.rstrip("/")
    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_FIGURE_SEARCH_SQL, {
                "vec": vec_str, "query": query, "top_k": top_k,
                "doc_type": document_type, "year_from": year_from, "year_to": year_to,
            })
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
