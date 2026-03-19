"""AWS Lambda handler: S3 event on processed JSONL → chunk + embed + index into Neon."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote_plus

import boto3
import psycopg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

_db_conn = None


def _get_db_conn():
    """Get or create a database connection for pipeline status tracking."""
    global _db_conn
    dsn = os.environ.get("CHUNKER_DATABASE_DSN")
    if not dsn:
        return None
    if _db_conn is None or _db_conn.closed:
        _db_conn = psycopg.connect(dsn, autocommit=True)
    return _db_conn


def _write_status(bucket, filename, stage, error=None):
    """Write pipeline status to the database (with S3 fallback)."""
    conn = _get_db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO pipeline_status (filename, stage, error, updated_at)
                       VALUES (%s, %s, %s, now())
                       ON CONFLICT (filename)
                       DO UPDATE SET stage = EXCLUDED.stage, error = EXCLUDED.error, updated_at = now()""",
                    (filename, stage, error),
                )
            return
        except Exception:
            logger.warning("DB status write failed, falling back to S3", exc_info=True)
            global _db_conn
            _db_conn = None

    # Fallback to S3
    status = {
        "filename": filename,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"status/{filename}.json",
        Body=json.dumps(status),
        ContentType="application/json",
    )


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

        s3_prefix = f"processed/{doc_id}/v{version}"

        # Read filename from _meta.json for status tracking
        filename = _read_meta_filename(bucket, f"{s3_prefix}/_meta.json")

        if filename:
            _write_status(bucket, filename, "indexing")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Download all JSONL files for this version
                prefix = f"{s3_prefix}/"
                _download_version(bucket, prefix, tmpdir, doc_id)

                # Download metadata overlay (lives outside version directory)
                doc_dir = tmpdir / "out" / doc_id
                overlay_key = f"processed/{doc_id}/documents_overlay.jsonl"
                try:
                    s3.download_file(bucket, overlay_key, str(doc_dir / "documents_overlay.jsonl"))
                    logger.info("Downloaded overlay: %s", overlay_key)
                except Exception:
                    pass  # No overlay is the normal case

                # Run chunk + embed + index
                _run_chunk_and_index(doc_id, tmpdir, version, s3_prefix=s3_prefix)
                logger.info("Indexing complete: doc_id=%s v%d", doc_id, version)

                if filename:
                    _write_status(bucket, filename, "done")
        except Exception:
            logger.exception("Failed to index %s v%d", doc_id, version)
            if filename:
                _write_status(bucket, filename, "error", "Indexing failed")

    return {"statusCode": 200, "body": json.dumps("OK")}


def _read_meta_filename(bucket: str, key: str) -> str | None:
    """Read filename from _meta.json in S3."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        meta = json.loads(resp["Body"].read())
        return meta.get("filename") or None
    except Exception:
        logger.warning("Could not read %s", key)
        return None


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


def _run_chunk_and_index(doc_id: str, data_dir: Path, version: int, *, s3_prefix: str = "") -> None:
    """Chunk, embed, and index a processed document into Neon."""
    from ras_chunker.config import ChunkerConfig
    from ras_chunker.pipeline import run_index

    config = ChunkerConfig(data_dir=data_dir)
    run_index(doc_id, config, s3_prefix=s3_prefix, version=version)
