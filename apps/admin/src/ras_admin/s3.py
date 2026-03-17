"""S3 operations for admin app."""

from __future__ import annotations

import json
import time
from datetime import UTC
from typing import Any


def get_client():
    import boto3

    return boto3.client("s3")


def list_versions(s3, bucket: str, doc_id: str) -> list[dict[str, Any]]:
    """List all version directories for a doc_id under processed/{doc_id}/v*/."""
    paginator = s3.get_paginator("list_objects_v2")
    versions = []
    seen = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=f"processed/{doc_id}/v", Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            p = prefix["Prefix"]  # e.g. processed/{doc_id}/v2/
            version_str = p.rstrip("/").rsplit("/", 1)[-1]  # e.g. v2
            if version_str.startswith("v") and version_str[1:].isdigit():
                v = int(version_str[1:])
                if v not in seen:
                    seen.add(v)
                    versions.append({"version": v, "prefix": p})

    versions.sort(key=lambda x: x["version"])
    return versions


def get_version_meta(s3, bucket: str, prefix: str) -> dict[str, Any] | None:
    """Read _meta.json from a version directory."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{prefix}_meta.json")
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def download_jsonl(s3, bucket: str, key: str) -> str | None:
    """Download a JSONL file from S3 as text."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read().decode("utf-8")
    except Exception:
        return None


def get_all_statuses(s3, bucket: str) -> list[dict[str, Any]]:
    """Get all pipeline status files from status/*.json."""
    paginator = s3.get_paginator("list_objects_v2")
    statuses = []
    for page in paginator.paginate(Bucket=bucket, Prefix="status/"):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                status = json.loads(resp["Body"].read())
                statuses.append(status)
            except Exception:
                continue
    return statuses


def get_status_for_file(s3, bucket: str, filename: str) -> dict[str, Any] | None:
    """Get pipeline status for a specific file."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"status/{filename}.json")
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def presign_upload(s3, bucket: str, filename: str) -> str:
    """Generate a presigned PUT URL for uploading a PDF."""
    import os

    safe_name = os.path.basename(filename)
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": f"uploads/{safe_name}", "ContentType": "application/pdf"},
        ExpiresIn=3600,
    )


def write_status(s3, bucket: str, filename: str, stage: str) -> None:
    """Write pipeline status for a file."""
    import os
    from datetime import datetime

    safe_name = os.path.basename(filename)
    status = {
        "filename": safe_name,
        "stage": stage,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"status/{safe_name}.json",
        Body=json.dumps(status),
        ContentType="application/json",
    )


def trigger_reprocess(s3, bucket: str, s3_key: str) -> None:
    """Trigger reprocessing by copying a file onto itself (fires S3 event)."""
    s3.copy_object(
        Bucket=bucket,
        Key=s3_key,
        CopySource={"Bucket": bucket, "Key": s3_key},
        MetadataDirective="REPLACE",
        ContentType="application/pdf",
    )


def trigger_reindex(s3, bucket: str, doc_id: str, version: int) -> None:
    """Trigger reindexing by copying the latest documents.jsonl."""
    key = f"processed/{doc_id}/v{version}/documents.jsonl"
    s3.copy_object(
        Bucket=bucket,
        Key=key,
        CopySource={"Bucket": bucket, "Key": key},
        MetadataDirective="REPLACE",
        ContentType="application/jsonl",
    )


def delete_doc_from_s3(s3, bucket: str, doc_id: str, source_filename: str) -> None:
    """Delete all S3 objects for a document (processed/, uploads/, status/)."""
    paginator = s3.get_paginator("list_objects_v2")

    # Delete processed/{doc_id}/*
    for page in paginator.paginate(Bucket=bucket, Prefix=f"processed/{doc_id}/"):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})

    # Delete uploads/{filename}
    try:
        s3.delete_object(Bucket=bucket, Key=f"uploads/{source_filename}")
    except Exception:
        pass

    # Delete status/{filename}.json
    try:
        s3.delete_object(Bucket=bucket, Key=f"status/{source_filename}.json")
    except Exception:
        pass


def list_uploaded_pdfs(s3, bucket: str) -> list[str]:
    """List all PDFs in uploads/."""
    paginator = s3.get_paginator("list_objects_v2")
    pdfs = []
    for page in paginator.paginate(Bucket=bucket, Prefix="uploads/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdfs.append(obj["Key"])
    return pdfs


def get_latest_version_prefix(s3, bucket: str, doc_id: str) -> str | None:
    """Return the S3 prefix for the latest version of a doc_id, e.g. 'processed/{doc_id}/v3/'."""
    versions = list_versions(s3, bucket, doc_id)
    if not versions:
        return None
    return versions[-1]["prefix"]


def download_jsonl_as_list(s3, bucket: str, key: str) -> list[dict[str, Any]] | None:
    """Download a JSONL file from S3 and return as a list of dicts."""
    text = download_jsonl(s3, bucket, key)
    if text is None:
        return None
    records = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def presign_get(s3, bucket: str, key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL for an S3 object."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def download_overlay(s3, bucket: str, doc_id: str) -> dict[str, Any] | None:
    """Read metadata overlay for a document. Returns parsed dict or None."""
    key = f"processed/{doc_id}/documents_overlay.jsonl"
    text = download_jsonl(s3, bucket, key)
    if text is None:
        return None
    first_line = text.strip().splitlines()[0]
    return json.loads(first_line)


def upload_overlay(s3, bucket: str, doc_id: str, overlay: dict[str, Any]) -> None:
    """Write metadata overlay for a document."""
    key = f"processed/{doc_id}/documents_overlay.jsonl"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(overlay, ensure_ascii=False) + "\n",
        ContentType="application/jsonl",
    )


def reprocess_all(s3, bucket: str) -> list[str]:
    """Trigger reprocessing for all uploaded PDFs. Returns filenames."""
    pdf_keys = list_uploaded_pdfs(s3, bucket)
    filenames = []
    for i, key in enumerate(pdf_keys):
        filename = key.rsplit("/", 1)[-1]
        write_status(s3, bucket, filename, "uploaded")
        filenames.append(filename)
        trigger_reprocess(s3, bucket, key)
        if i < len(pdf_keys) - 1:
            time.sleep(0.1)
    return filenames
