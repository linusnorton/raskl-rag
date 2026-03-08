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

        s3_prefix = f"processed/{doc_id}/v{version}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download all JSONL files for this version
            prefix = f"{s3_prefix}/"
            _download_version(bucket, prefix, tmpdir, doc_id)

            # Run chunk + embed + index
            _run_chunk_and_index(doc_id, tmpdir, version, s3_prefix=s3_prefix)
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


def _run_chunk_and_index(doc_id: str, data_dir: Path, version: int, *, s3_prefix: str = "") -> None:
    """Chunk, embed, and index a processed document into Neon."""
    from ras_chunker.config import ChunkerConfig
    from ras_chunker.pipeline import run_index

    config = ChunkerConfig(data_dir=data_dir)
    run_index(doc_id, config, s3_prefix=s3_prefix, version=version)
