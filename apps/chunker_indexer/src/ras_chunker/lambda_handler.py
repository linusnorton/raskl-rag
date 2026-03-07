"""AWS Lambda handler: S3 event on processed JSONL → chunk + embed + index into Neon."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote_plus

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """Process S3 event: download JSONL, chunk + embed + index into Neon."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        # Only trigger on documents.jsonl
        if not key.endswith("documents.jsonl"):
            logger.info("Skipping non-documents.jsonl: %s", key)
            continue

        # Extract doc_id and version from key: processed/{doc_id}/v{N}/documents.jsonl
        m = re.match(r"^processed/([^/]+)/v(\d+)/documents\.jsonl$", key)
        if not m:
            logger.warning("Unexpected key format: %s", key)
            continue

        doc_id = m.group(1)
        version = int(m.group(2))

        logger.info("Indexing s3://%s/%s (doc_id=%s, version=%d)", bucket, key, doc_id, version)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download all JSONL files for this version
            prefix = f"processed/{doc_id}/v{version}/"
            _download_version(bucket, prefix, tmpdir, doc_id)

            # Run chunk + embed + index
            _run_chunk_and_index(doc_id, tmpdir, version)
            logger.info("Indexing complete: doc_id=%s v%d", doc_id, version)

    return {"statusCode": 200, "body": json.dumps("OK")}


def _download_version(bucket: str, prefix: str, tmpdir: Path, doc_id: str) -> None:
    """Download all files from a versioned S3 prefix into a local doc_id directory structure."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    # Create the expected directory structure: {tmpdir}/out/{doc_id}/
    doc_dir = tmpdir / "out" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.rsplit("/", 1)[-1]
        if filename.startswith("_"):
            continue  # Skip _meta.json
        local_path = doc_dir / filename
        s3.download_file(bucket, key, str(local_path))
        logger.info("Downloaded %s", key)


def _run_chunk_and_index(doc_id: str, data_dir: Path, version: int) -> None:
    """Chunk, embed, and index a processed document into Neon."""
    from ras_chunker.config import ChunkerConfig
    from ras_chunker.db import get_connection, upsert_chunks, upsert_document
    from ras_chunker.embedder import embed_chunks
    from ras_chunker.pipeline import load_and_chunk

    config = ChunkerConfig(data_dir=data_dir)
    output, chunks = load_and_chunk(doc_id, config)

    if not chunks:
        logger.warning(
            "No chunks produced for %s — skipping DB update (stale chunks may remain)", doc_id
        )
        return

    embeddings = embed_chunks(chunks, config)
    logger.info("Embedded %d chunks", len(embeddings))

    with get_connection(config) as conn:
        upsert_document(conn, output.meta)
        upsert_chunks(conn, chunks, embeddings, doc_id=doc_id, version=version)
        conn.commit()

    logger.info("Indexed %d chunks for %s v%d", len(chunks), doc_id, version)
