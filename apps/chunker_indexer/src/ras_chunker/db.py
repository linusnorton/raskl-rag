"""PostgreSQL/pgvector database operations."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from .config import ChunkerConfig
from .schema import Chunk, DocMeta, FigureMeta

DDL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    title TEXT,
    author TEXT,
    editor TEXT,
    year INTEGER,
    publication TEXT,
    document_type TEXT,
    abstract TEXT,
    keywords TEXT[] NOT NULL DEFAULT '{}',
    language TEXT,
    isbn TEXT,
    issn TEXT,
    series TEXT,
    description TEXT,
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

-- Migration: add s3_prefix to existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS s3_prefix TEXT NOT NULL DEFAULT '';

-- Migration: add document_type to existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type TEXT;

-- Migration: add publication to existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS publication TEXT;

-- Migration: add enriched metadata columns
ALTER TABLE documents ADD COLUMN IF NOT EXISTS editor TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS abstract TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS keywords TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS isbn TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS issn TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS series TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS description TEXT;

CREATE TABLE IF NOT EXISTS figures (
    figure_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_num INTEGER NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    asset_path TEXT NOT NULL,
    thumb_path TEXT NOT NULL DEFAULT '',
    embedding vector(1024),
    indexed_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_figures_doc_id ON figures(doc_id);
CREATE INDEX IF NOT EXISTS idx_figures_embedding_hnsw
    ON figures USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_figures_caption_fts
    ON figures USING gin(to_tsvector('english', caption));
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
        INSERT INTO documents (doc_id, source_filename, title, author, editor, year, publication,
                               document_type, abstract, keywords, language, isbn, issn, series,
                               description, page_offset, sha256_pdf, s3_prefix)
        VALUES (%(doc_id)s, %(source_filename)s, %(title)s, %(author)s, %(editor)s, %(year)s,
                %(publication)s, %(document_type)s, %(abstract)s, %(keywords)s, %(language)s,
                %(isbn)s, %(issn)s, %(series)s, %(description)s, %(page_offset)s, %(sha256_pdf)s,
                %(s3_prefix)s)
        ON CONFLICT (doc_id) DO UPDATE SET
            source_filename = EXCLUDED.source_filename,
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            editor = EXCLUDED.editor,
            year = EXCLUDED.year,
            publication = EXCLUDED.publication,
            document_type = EXCLUDED.document_type,
            abstract = EXCLUDED.abstract,
            keywords = EXCLUDED.keywords,
            language = EXCLUDED.language,
            isbn = EXCLUDED.isbn,
            issn = EXCLUDED.issn,
            series = EXCLUDED.series,
            description = EXCLUDED.description,
            page_offset = EXCLUDED.page_offset,
            sha256_pdf = EXCLUDED.sha256_pdf,
            s3_prefix = EXCLUDED.s3_prefix,
            indexed_at = now()
        """,
        meta.model_dump(),
    )


def get_existing_chunks(conn: psycopg.Connection, doc_id: str) -> dict[str, tuple[str, list[float]]]:
    """Fetch existing chunk_id → (text, embedding) for a document."""
    result: dict[str, tuple[str, list[float]]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text, embedding FROM chunks WHERE doc_id = %s", (doc_id,))
        for row in cur.fetchall():
            result[row[0]] = (row[1], list(row[2]))
    return result


def get_existing_figures(conn: psycopg.Connection, doc_id: str) -> dict[str, tuple[str, list[float] | None]]:
    """Fetch existing figure_id → (caption, embedding) for a document."""
    result: dict[str, tuple[str, list[float] | None]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT figure_id, caption, embedding FROM figures WHERE doc_id = %s", (doc_id,))
        for row in cur.fetchall():
            result[row[0]] = (row[1], list(row[2]) if row[2] is not None else None)
    return result


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


def upsert_figures(
    conn: psycopg.Connection,
    figures: list[FigureMeta],
    embeddings: list[list[float] | None],
) -> None:
    """Insert or update figures with their embeddings."""
    if not figures:
        return

    doc_id = figures[0].doc_id
    conn.execute("DELETE FROM figures WHERE doc_id = %s", (doc_id,))

    with conn.cursor() as cur:
        for fig, embedding in zip(figures, embeddings):
            cur.execute(
                """
                INSERT INTO figures (figure_id, doc_id, page_num, caption, asset_path, thumb_path, embedding)
                VALUES (%(figure_id)s, %(doc_id)s, %(page_num)s, %(caption)s, %(asset_path)s, %(thumb_path)s, %(embedding)s)
                """,
                {**fig.model_dump(), "embedding": embedding},
            )


def get_figures_for_pages(conn: psycopg.Connection, doc_id_pages: list[tuple[str, int]]) -> list[dict]:
    """Retrieve figures matching a list of (doc_id, page_num) pairs."""
    if not doc_id_pages:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.figure_id, f.doc_id, f.page_num, f.caption, f.asset_path, f.thumb_path,
                   d.source_filename
            FROM figures f
            JOIN documents d ON f.doc_id = d.doc_id
            WHERE (f.doc_id, f.page_num) IN (SELECT unnest(%(doc_ids)s::text[]), unnest(%(pages)s::int[]))
            """,
            {
                "doc_ids": [dp[0] for dp in doc_id_pages],
                "pages": [dp[1] for dp in doc_id_pages],
            },
        )
        rows = cur.fetchall()

    return [
        {
            "figure_id": r[0],
            "doc_id": r[1],
            "page_num": r[2],
            "caption": r[3],
            "asset_path": r[4],
            "thumb_path": r[5],
            "source_filename": r[6],
        }
        for r in rows
    ]
