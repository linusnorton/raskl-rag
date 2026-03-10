"""AWS Lambda handler: S3 event → download PDF → Qwen3 VL OCR → pipeline → versioned S3 output → diff + email."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote_plus

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def _cleanup_tmp():
    """Remove stale temp files from previous invocations that timed out.

    When a Lambda times out, the TemporaryDirectory context manager never
    runs cleanup. The next invocation on the same warm container inherits
    the leftover files, which can fill /tmp and cause 'No usable temporary
    directory found' errors.
    """
    tmp = Path("/tmp")
    for entry in tmp.iterdir():
        if entry.name in ("uv-cache",):
            continue  # preserve uv cache
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except Exception:
            pass


def _write_status(bucket, filename, stage, error=None):
    """Write pipeline status for a file to S3."""
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
    """Process S3 event: download PDF, run docproc with Qwen3 VL, upload versioned JSONL, diff + email."""
    _cleanup_tmp()

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        if not key.lower().endswith(".pdf"):
            logger.info("Skipping non-PDF: %s", key)
            continue

        filename = os.path.basename(key)
        logger.info("Processing s3://%s/%s", bucket, key)
        _write_status(bucket, filename, "processing")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Download PDF
                pdf_path = tmpdir / filename
                s3.download_file(bucket, key, str(pdf_path))
                logger.info("Downloaded PDF: %s (%d bytes)", pdf_path.name, pdf_path.stat().st_size)

                # Run docproc pipeline (Qwen3 VL via Bedrock)
                out_dir = tmpdir / "data"
                doc_id = _run_docproc(pdf_path, out_dir)
                logger.info("Docproc complete: doc_id=%s", doc_id)

                # Guard: don't upload empty output (e.g. all pages failed OCR due to throttling)
                text_blocks_file = out_dir / "out" / doc_id / "text_blocks.jsonl"
                if not text_blocks_file.exists() or text_blocks_file.stat().st_size == 0:
                    logger.error(
                        "Aborting upload for %s: text_blocks.jsonl is empty — OCR likely failed on all pages",
                        doc_id,
                    )
                    _write_status(bucket, filename, "error", "OCR failed — empty output")
                    continue

                # Upload versioned JSONL to S3 and run diff + email
                version = _upload_versioned_output(bucket, doc_id, out_dir, tmpdir, filename=filename)
                logger.info("Output uploaded to s3://%s/processed/%s/v%d/", bucket, doc_id, version)
                _write_status(bucket, filename, "processed")
        except Exception:
            logger.exception("Failed to process %s", filename)
            _write_status(bucket, filename, "error", "DocProc failed")

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
    )
    return doc_id


def _get_next_version(bucket: str, doc_id: str) -> int:
    """Find the next version number by listing existing versions in S3."""
    prefix = f"processed/{doc_id}/v"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

    max_version = 0
    for cp in response.get("CommonPrefixes", []):
        # Extract version number from prefix like "processed/{doc_id}/v3/"
        m = re.search(r"/v(\d+)/$", cp["Prefix"])
        if m:
            max_version = max(max_version, int(m.group(1)))

    return max_version + 1


def _download_previous_version(bucket: str, doc_id: str, version: int, local_dir: Path) -> Path | None:
    """Download previous version's JSONL files to a local directory. Returns local dir or None."""
    if version <= 1:
        return None

    prev_version = version - 1
    prefix = f"processed/{doc_id}/v{prev_version}/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if response.get("KeyCount", 0) == 0:
        return None

    prev_dir = local_dir / f"prev_v{prev_version}"
    prev_dir.mkdir(parents=True, exist_ok=True)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.rsplit("/", 1)[-1]
        s3.download_file(bucket, key, str(prev_dir / filename))

    return prev_dir


def _upload_versioned_output(bucket: str, doc_id: str, data_dir: Path, tmpdir: Path, *, filename: str = "") -> int:
    """Upload processed JSONL files to S3 under processed/{doc_id}/v{N}/, run diff + email."""
    doc_dir = data_dir / "out" / doc_id

    if not doc_dir.is_dir():
        logger.warning("No output directory for %s", doc_id)
        return 0

    version = _get_next_version(bucket, doc_id)
    s3_prefix = f"processed/{doc_id}/v{version}/"

    # Upload all JSONL files
    for f in doc_dir.iterdir():
        if f.is_file():
            s3_key = f"{s3_prefix}{f.name}"
            s3.upload_file(str(f), bucket, s3_key)
            logger.info("Uploaded %s", s3_key)

    # Upload assets (images, thumbnails)
    assets_dir = doc_dir / "assets"
    if assets_dir.is_dir():
        for asset in assets_dir.iterdir():
            if asset.is_file():
                s3_key = f"{s3_prefix}assets/{asset.name}"
                content_type = "image/jpeg" if asset.suffix in (".jpg", ".jpeg") else "image/png"
                s3.upload_file(str(asset), bucket, s3_key, ExtraArgs={"ContentType": content_type})
                logger.info("Uploaded asset %s", s3_key)

    # Write version metadata
    meta = {
        "version": version,
        "backend": "qwen3vl",
        "doc_id": doc_id,
        "filename": filename,
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"{s3_prefix}_meta.json",
        Body=json.dumps(meta, default=str),
        ContentType="application/json",
    )

    # Diff against previous version and email
    _diff_and_notify(bucket, doc_id, version, doc_dir, tmpdir)

    return version


def _diff_and_notify(bucket: str, doc_id: str, version: int, new_dir: Path, tmpdir: Path) -> None:
    """Download previous version, diff, and send email notification."""
    from ras_docproc.diff import diff_versions
    from ras_docproc.notify import send_diff_email

    prev_dir = _download_previous_version(bucket, doc_id, version, tmpdir)
    if prev_dir is None:
        logger.info("No previous version for %s — skipping diff", doc_id)
        return

    report = diff_versions(prev_dir, new_dir, doc_id, version - 1, version)
    logger.info(
        "Diff %s v%d→v%d: +%d -%d ~%d =%d",
        doc_id,
        version - 1,
        version,
        report.blocks_added,
        report.blocks_removed,
        report.blocks_changed,
        report.blocks_unchanged,
    )

    try:
        region = os.environ.get("DOCPROC_BEDROCK_REGION", "eu-west-2")
        send_diff_email(report, region=region)
    except Exception:
        logger.exception("Failed to send diff email for %s v%d", doc_id, version)
