"""One-shot Lambda: initialize Neon database schema (pgvector + tables + indexes)."""

import os

import psycopg2


def lambda_handler(event, context):
    dsn = os.environ["DATABASE_DSN"]
    embed_dims = int(os.environ.get("EMBED_DIMENSIONS", "1024"))

    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        doc_id          TEXT PRIMARY KEY,
        source_filename TEXT NOT NULL,
        title           TEXT,
        author          TEXT,
        year            INTEGER,
        page_offset     INTEGER NOT NULL DEFAULT 0,
        sha256_pdf      TEXT NOT NULL,
        indexed_at      TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id        TEXT PRIMARY KEY,
        doc_id          TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
        chunk_index     INTEGER NOT NULL,
        start_page      INTEGER NOT NULL,
        end_page        INTEGER NOT NULL,
        section_heading TEXT,
        text            TEXT NOT NULL,
        block_ids       TEXT[] NOT NULL,
        token_count     INTEGER NOT NULL,
        embedding       vector({embed_dims}) NOT NULL,
        indexed_at      TIMESTAMPTZ DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
        ON chunks(doc_id);

    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);

    CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
        ON chunks USING gin (to_tsvector('english', text));

    CREATE TABLE IF NOT EXISTS chunk_changes (
        id              SERIAL PRIMARY KEY,
        doc_id          TEXT NOT NULL,
        version         INTEGER NOT NULL,
        indexed_at      TIMESTAMPTZ DEFAULT now(),
        chunks_total    INTEGER NOT NULL,
        chunks_added    INTEGER NOT NULL DEFAULT 0,
        chunks_removed  INTEGER NOT NULL DEFAULT 0,
        chunks_text_changed INTEGER NOT NULL DEFAULT 0,
        chunks_unchanged INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_chunk_changes_doc ON chunk_changes(doc_id);
    """

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        return {"statusCode": 200, "body": "Schema initialized successfully"}
    finally:
        conn.close()
