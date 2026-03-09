"""Upload Lambda: multi-file upload with presigned URLs, reprocess all, and pipeline status tracking."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3

s3 = boto3.client("s3")
BUCKET = os.environ["DOCS_BUCKET"]
PASSWORD_HASH = os.environ["UPLOAD_PASSWORD_HASH"]

HTML_TEMPLATE = (Path(__file__).parent / "template.html").read_text()


def _verify_password(password: str) -> bool:
    """Verify password against SHA-256 hash."""
    h = hashlib.sha256(password.encode()).hexdigest()
    return h == PASSWORD_HASH


def _json_response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _html_response(status):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "text/html; charset=utf-8"},
        "body": HTML_TEMPLATE,
    }


def _write_status(filename, stage, error=None):
    """Write pipeline status for a file to S3."""
    status = {
        "filename": filename,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"status/{filename}.json",
        Body=json.dumps(status),
        ContentType="application/json",
    )


def _handle_status(event):
    """GET /upload/status — return JSON array of all file statuses."""
    password = event.get("queryStringParameters", {}).get("password", "")
    if not _verify_password(password):
        return _json_response(403, {"error": "Invalid password"})

    paginator = s3.get_paginator("list_objects_v2")
    statuses = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="status/"):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
                status = json.loads(resp["Body"].read())
                statuses.append(status)
            except Exception:
                continue

    return _json_response(200, statuses)


def _handle_presign(event):
    """POST /upload/presign — return a presigned S3 PUT URL for large files."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return _json_response(400, {"error": "Invalid JSON body"})

    password = body.get("password", "")
    if not _verify_password(password):
        return _json_response(403, {"error": "Invalid password"})

    filename = body.get("filename", "")
    if not filename or not filename.lower().endswith(".pdf"):
        return _json_response(400, {"error": "filename must end with .pdf"})

    safe_name = os.path.basename(filename)
    s3_key = f"uploads/{safe_name}"

    presigned_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": s3_key, "ContentType": "application/pdf"},
        ExpiresIn=600,
    )

    _write_status(safe_name, "uploaded")

    return _json_response(200, {"upload_url": presigned_url, "s3_key": s3_key})


def _handle_reprocess_all(event):
    """POST /upload/reprocess — re-trigger pipeline for all PDFs, return filenames as JSON."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return _json_response(400, {"error": "Invalid JSON body"})

    password = body.get("password", "")
    if not _verify_password(password):
        return _json_response(403, {"error": "Invalid password"})

    paginator = s3.get_paginator("list_objects_v2")
    pdf_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="uploads/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    if not pdf_keys:
        return _json_response(200, {"filenames": [], "message": "No PDFs found in uploads/"})

    import time

    filenames = []
    for i, key in enumerate(pdf_keys):
        filename = key.rsplit("/", 1)[-1]
        _write_status(filename, "uploaded")
        filenames.append(filename)

        s3.copy_object(
            Bucket=BUCKET,
            Key=key,
            CopySource={"Bucket": BUCKET, "Key": key},
            MetadataDirective="REPLACE",
            ContentType="application/pdf",
        )
        # Stagger S3 events to avoid overwhelming Bedrock with concurrent DocProc Lambdas
        if i < len(pdf_keys) - 1:
            time.sleep(0.1)

    return _json_response(200, {"filenames": filenames, "message": f"Reprocessing {len(filenames)} PDFs"})


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("requestContext", {}).get("http", {}).get("path", "/")

    # Presigned URL endpoint
    if path.endswith("/presign") and method == "POST":
        return _handle_presign(event)

    # Status endpoint
    if path.endswith("/status") and method == "GET":
        return _handle_status(event)

    # Reprocess endpoint
    if path.endswith("/reprocess") and method == "POST":
        return _handle_reprocess_all(event)

    # Serve HTML page
    if method == "GET":
        return _html_response(200)

    return _json_response(400, {"error": "Unknown request"})
