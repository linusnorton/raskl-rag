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

CREATE TABLE IF NOT EXISTS chunk_changes (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now(),
    chunks_total INTEGER NOT NULL,
    chunks_added INTEGER NOT NULL DEFAULT 0,
    chunks_removed INTEGER NOT NULL DEFAULT 0,
    chunks_text_changed INTEGER NOT NULL DEFAULT 0,
    chunks_unchanged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunk_changes_doc ON chunk_changes(doc_id);
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


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    doc_id: str | None = None,
    version: int = 0,
) -> None:
    """Insert or update chunks with their embeddings, logging change audit."""
    if not chunks:
        return

    chunk_doc_id = doc_id or chunks[0].doc_id

    # Snapshot old chunks for change tracking
    old_chunks: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text FROM chunks WHERE doc_id = %s", (chunk_doc_id,))
        for row in cur.fetchall():
            old_chunks[row[0]] = row[1]

    # Build new chunk map
    new_chunks = {chunk.chunk_id: chunk.text for chunk in chunks}

    old_ids = set(old_chunks.keys())
    new_ids = set(new_chunks.keys())
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)
    text_changed = sum(1 for cid in (old_ids & new_ids) if old_chunks[cid] != new_chunks[cid])
    unchanged = sum(1 for cid in (old_ids & new_ids) if old_chunks[cid] == new_chunks[cid])

    # Delete and re-insert
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (chunk_doc_id,))

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

        # Log change audit
        cur.execute(
            """
            INSERT INTO chunk_changes (doc_id, version, chunks_total, chunks_added, chunks_removed,
                                       chunks_text_changed, chunks_unchanged)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (chunk_doc_id, version, len(chunks), added, removed, text_changed, unchanged),
        )
