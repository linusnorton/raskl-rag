"""RAG retrieval: embed query → pgvector cosine search."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from .config import ChatConfig
from .providers import get_embed_provider

log = logging.getLogger(__name__)


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
    page_offset: int = 0


def embed_query(query: str, config: ChatConfig) -> list[float]:
    """Embed a single query string via the configured provider."""
    provider = get_embed_provider(config)
    embeddings = provider.embed([query])
    return embeddings[0]


RETRIEVE_SQL = """\
WITH vector_results AS (
    SELECT chunk_id, doc_id, text, start_page, end_page, section_heading,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS vrank
    FROM chunks
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT 50
),
or_query AS (
    SELECT to_tsquery('english',
           replace(plainto_tsquery('english', %(query)s)::text, '&', '|')
    ) AS q
),
text_results AS (
    SELECT c.chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(
               setweight(to_tsvector('english', c.text), 'B')
               || setweight(to_tsvector('english',
                   coalesce(d.title, '') || ' ' || coalesce(d.author, '')), 'A'),
               oq.q) DESC) AS trank
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id, or_query oq
    WHERE to_tsvector('english', c.text) @@ oq.q
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
       d.source_filename, d.title, d.author, d.year, d.page_offset, f.rrf_score AS score
FROM fused f
JOIN documents d ON f.doc_id = d.doc_id
ORDER BY f.rrf_score DESC
LIMIT %(top_k)s
"""


def retrieve(query: str, config: ChatConfig, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed the query and retrieve the most similar chunks via hybrid search (vector + full-text RRF)."""
    top_k = top_k or config.retrieval_top_k
    fetch_k = config.rerank_candidates if config.rerank_enabled else top_k
    vec = embed_query(query, config)
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(RETRIEVE_SQL, {"vec": vec_str, "query": query, "top_k": fetch_k})
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
            score=row[11],
        ))

    if config.rerank_enabled and chunks:
        from .reranker import rerank
        chunks = rerank(query, chunks, config, top_k=top_k)

    return chunks
