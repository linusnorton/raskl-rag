"""Migrate chunks from local PostgreSQL to Neon, re-embedding with Bedrock Titan Embed v2.

Reads chunk text + metadata from the local DB (known-good chunks), re-embeds with
Bedrock Titan Embed v2, and upserts into Neon. Also migrates document metadata.

Requires:
- Local PostgreSQL running (docker compose up)
- AWS credentials with Bedrock access
- NEON_DSN environment variable pointing to the Neon database

Usage:
    # Dry run (count chunks, no writes)
    uv run python scripts/migrate_to_neon.py --dry-run

    # Migrate all documents
    uv run python scripts/migrate_to_neon.py --neon-dsn "postgresql://..."

    # Migrate a single document
    uv run python scripts/migrate_to_neon.py --neon-dsn "postgresql://..." --doc-id swettenham-journal-1874-1876-bbfb9df1239d
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
from pgvector.psycopg import register_vector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Bedrock Titan Embed v2 configuration
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_REGION = "eu-west-2"
EMBED_DIMENSIONS = 1024
EMBED_BATCH_CONCURRENCY = 10


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using Bedrock Titan Embed v2 with concurrent requests."""
    import boto3

    def _embed_one(text: str) -> list[float]:
        client = boto3.client("bedrock-runtime", region_name=EMBED_REGION)
        body = json.dumps({
            "inputText": text,
            "dimensions": EMBED_DIMENSIONS,
            "normalize": True,
        })
        resp = client.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["embedding"]

    with ThreadPoolExecutor(max_workers=EMBED_BATCH_CONCURRENCY) as pool:
        return list(pool.map(_embed_one, texts))


def fetch_documents(conn: psycopg.Connection, doc_id: str | None = None) -> list[dict]:
    """Fetch document metadata from local DB."""
    sql = "SELECT doc_id, source_filename, title, author, year, page_offset, sha256_pdf FROM documents"
    params: dict = {}
    if doc_id:
        sql += " WHERE doc_id = %(doc_id)s"
        params["doc_id"] = doc_id
    sql += " ORDER BY doc_id"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_chunks(conn: psycopg.Connection, doc_id: str) -> list[dict]:
    """Fetch all chunks for a document from local DB (without embeddings)."""
    sql = """
        SELECT chunk_id, doc_id, chunk_index, start_page, end_page,
               section_heading, text, block_ids, token_count
        FROM chunks
        WHERE doc_id = %(doc_id)s
        ORDER BY chunk_index
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"doc_id": doc_id})
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def upsert_document(conn: psycopg.Connection, doc: dict) -> None:
    """Upsert document metadata into Neon."""
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
        doc,
    )


def upsert_chunks(conn: psycopg.Connection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Delete existing chunks for the doc and insert new ones into Neon."""
    if not chunks:
        return

    doc_id = chunks[0]["doc_id"]
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, start_page, end_page,
                                    section_heading, text, block_ids, token_count, embedding)
                VALUES (%(chunk_id)s, %(doc_id)s, %(chunk_index)s, %(start_page)s, %(end_page)s,
                        %(section_heading)s, %(text)s, %(block_ids)s, %(token_count)s, %(embedding)s)
                """,
                {**chunk, "embedding": embedding},
            )


def main():
    parser = argparse.ArgumentParser(description="Migrate local DB chunks to Neon with Bedrock embeddings")
    parser.add_argument("--neon-dsn", required=False, help="Neon database DSN")
    parser.add_argument("--local-db", default="raskl_rag", help="Local database name (default: raskl_rag)")
    parser.add_argument("--doc-id", help="Migrate only this document (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size (default: 50)")
    args = parser.parse_args()

    if not args.dry_run and not args.neon_dsn:
        parser.error("--neon-dsn is required unless using --dry-run")

    local_dsn = f"postgresql://raskl:raskl@localhost:5432/{args.local_db}"
    log.info("Connecting to local DB: %s", args.local_db)

    with psycopg.connect(local_dsn) as local_conn:
        register_vector(local_conn)
        documents = fetch_documents(local_conn, args.doc_id)
        log.info("Found %d document(s) to migrate", len(documents))

        if not documents:
            log.warning("No documents found")
            return

        total_chunks = 0
        for doc in documents:
            chunks = fetch_chunks(local_conn, doc["doc_id"])
            total_chunks += len(chunks)
            log.info("  %s: %d chunks", doc["doc_id"], len(chunks))

        if args.dry_run:
            log.info("DRY RUN: would migrate %d documents, %d chunks total", len(documents), total_chunks)
            return

    with psycopg.connect(local_dsn) as local_conn:
        register_vector(local_conn)
        migrated_docs = 0
        migrated_chunks = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            log.info("Migrating %s ...", doc_id)

            chunks = fetch_chunks(local_conn, doc_id)
            if not chunks:
                log.warning("  No chunks for %s, skipping", doc_id)
                continue

            # Re-embed in batches
            all_embeddings: list[list[float]] = []
            texts = [c["text"] for c in chunks]

            for i in range(0, len(texts), args.batch_size):
                batch = texts[i : i + args.batch_size]
                t0 = time.time()
                embeddings = embed_texts(batch)
                dt = time.time() - t0
                all_embeddings.extend(embeddings)
                log.info("  Embedded %d/%d chunks (%.1fs)", min(i + args.batch_size, len(texts)), len(texts), dt)

            # Upsert into Neon (fresh connection per document to avoid timeouts)
            neon_conn = psycopg.connect(args.neon_dsn, sslmode="require")
            register_vector(neon_conn)
            try:
                with neon_conn:
                    upsert_document(neon_conn, doc)
                    upsert_chunks(neon_conn, chunks, all_embeddings)
                    neon_conn.commit()
            finally:
                neon_conn.close()

            migrated_docs += 1
            migrated_chunks += len(chunks)
            log.info("  Done: %d chunks migrated for %s", len(chunks), doc_id)

    log.info("Migration complete: %d documents, %d chunks", migrated_docs, migrated_chunks)


if __name__ == "__main__":
    main()
