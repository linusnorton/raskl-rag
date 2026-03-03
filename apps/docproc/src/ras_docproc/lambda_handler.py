"""AWS Lambda handler: S3 event → download PDF → docproc (Docling) → chunk → embed → index (Neon)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """Process S3 event: download PDF, run docproc + chunk + embed + index."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        if not key.lower().endswith(".pdf"):
            logger.info("Skipping non-PDF: %s", key)
            continue

        logger.info("Processing s3://%s/%s", bucket, key)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download PDF
            pdf_path = tmpdir / os.path.basename(key)
            s3.download_file(bucket, key, str(pdf_path))
            logger.info("Downloaded PDF: %s (%d bytes)", pdf_path.name, pdf_path.stat().st_size)

            # Run docproc pipeline (Docling only — no GPU for DeepSeek in Lambda)
            out_dir = tmpdir / "data"
            doc_id = _run_docproc(pdf_path, out_dir)
            logger.info("Docproc complete: doc_id=%s", doc_id)

            # Run chunk + embed + index
            _run_chunk_and_index(doc_id, out_dir)
            logger.info("Indexing complete: doc_id=%s", doc_id)

            # Upload processed JSONL to S3
            _upload_output(bucket, doc_id, out_dir)
            logger.info("Output uploaded to s3://%s/processed/%s/", bucket, doc_id)

    return {"statusCode": 200, "body": json.dumps("OK")}


def _run_docproc(pdf_path: Path, out_dir: Path) -> str:
    """Run the docproc pipeline on a PDF, return doc_id."""
    from ras_docproc.cli import _run_pipeline

    doc_id = _run_pipeline(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=None,
        page_range=None,
        force=True,
        backend="docling",
    )
    return doc_id


def _run_chunk_and_index(doc_id: str, data_dir: Path) -> None:
    """Chunk, embed, and index a processed document into Neon."""
    from ras_chunker.config import ChunkerConfig
    from ras_chunker.db import get_connection, upsert_chunks, upsert_document
    from ras_chunker.embedder import embed_chunks
    from ras_chunker.pipeline import load_and_chunk

    config = ChunkerConfig(data_dir=data_dir)
    output, chunks = load_and_chunk(doc_id, config)

    if not chunks:
        logger.warning("No chunks produced for %s", doc_id)
        return

    embeddings = embed_chunks(chunks, config)
    logger.info("Embedded %d chunks", len(embeddings))

    with get_connection(config) as conn:
        upsert_document(conn, output.meta)
        upsert_chunks(conn, chunks, embeddings)
        conn.commit()

    logger.info("Indexed %d chunks for %s", len(chunks), doc_id)


def _upload_output(bucket: str, doc_id: str, data_dir: Path) -> None:
    """Upload processed JSONL files to S3 under processed/{doc_id}/."""
    doc_dir = data_dir / "out" / doc_id

    if not doc_dir.is_dir():
        logger.warning("No output directory for %s", doc_id)
        return

    for f in doc_dir.iterdir():
        if f.is_file():
            s3_key = f"processed/{doc_id}/{f.name}"
            s3.upload_file(str(f), bucket, s3_key)
            logger.info("Uploaded %s", s3_key)
