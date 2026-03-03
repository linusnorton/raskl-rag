"""PostgreSQL/pgvector database operations."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from .config import ChunkerConfig
from .schema import Chunk, DocMeta

DDL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    title TEXT,
    author TEXT,
    year INTEGER,
    page_offset INTEGER NOT NULL DEFAULT 0,
    sha256_pdf TEXT NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    section_heading TEXT,
    text TEXT NOT NULL,
    block_ids TEXT[] NOT NULL,
    token_count INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON chunks USING gin(to_tsvector('english', text));

-- Migration: add page_offset to existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_offset INTEGER NOT NULL DEFAULT 0;
"""


def get_connection(config: ChunkerConfig) -> psycopg.Connection:
    conn = psycopg.connect(config.dsn, sslmode="prefer")
    register_vector(conn)
    return conn


def init_schema(config: ChunkerConfig) -> None:
    """Create tables and indexes."""
    conn = psycopg.connect(config.dsn, sslmode="prefer")
    with conn:
        conn.execute(DDL)
        conn.commit()
        register_vector(conn)
    conn.close()


def upsert_document(conn: psycopg.Connection, meta: DocMeta) -> None:
    """Insert or update document metadata."""
    conn.execute(
        """
        INSERT INTO documents (doc_id, source_filename, title, author, year, page_offset, sha256_pdf)
        VALUES (%(doc_id)s, %(source_filename)s, %(title)s, %(author)s, %(year)s, %(page_offset)s, %(sha256_pdf)s)
        ON CONFLICT (doc_id) DO UPDATE SET
            source_filename = EXCLUDED.source_filename,
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            year = EXCLUDED.year,
            page_offset = EXCLUDED.page_offset,
            sha256_pdf = EXCLUDED.sha256_pdf,
            indexed_at = now()
        """,
        meta.model_dump(),
    )


def upsert_chunks(conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Insert or update chunks with their embeddings."""
    # Delete existing chunks for this doc_id (re-index)
    if chunks:
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (chunks[0].doc_id,))

    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, start_page, end_page,
                                    section_heading, text, block_ids, token_count, embedding)
                VALUES (%(chunk_id)s, %(doc_id)s, %(chunk_index)s, %(start_page)s, %(end_page)s,
                        %(section_heading)s, %(text)s, %(block_ids)s, %(token_count)s, %(embedding)s)
                """,
                {**chunk.model_dump(), "embedding": embedding},
            )
