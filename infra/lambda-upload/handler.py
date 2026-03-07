"""Upload Lambda: HTML form with password validation + S3 presigned URL upload."""

import base64
import hashlib
import html
import json
import os
import time
import urllib.parse

import boto3

s3 = boto3.client("s3")
BUCKET = os.environ["DOCS_BUCKET"]
PASSWORD_HASH = os.environ["UPLOAD_PASSWORD_HASH"]

HTML_FORM = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwetBot Upload</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 480px; margin: 60px auto; padding: 0 20px; }
    h1 { font-size: 1.4em; }
    label { display: block; margin: 16px 0 4px; font-weight: 600; }
    input { width: 100%%; padding: 8px; box-sizing: border-box; }
    button { margin-top: 20px; padding: 10px 24px; cursor: pointer; }
    .msg { margin-top: 16px; padding: 12px; border-radius: 4px; }
    .ok { background: #d4edda; color: #155724; }
    .err { background: #f8d7da; color: #721c24; }
    .info { background: #cce5ff; color: #004085; }
    hr { margin: 24px 0; border: none; border-top: 1px solid #ddd; }
  </style>
</head>
<body>
  <h1>SwetBot PDF Upload</h1>
  <form method="post" enctype="multipart/form-data">
    <label>Password</label>
    <input type="password" name="password" required>
    <label>PDF file</label>
    <input type="file" name="file" accept=".pdf" required>
    <button type="submit">Upload</button>
  </form>
  <hr>
  <h2 style="font-size:1.1em;">Reprocess All Documents</h2>
  <p style="font-size:0.9em; color:#666;">Re-runs the full pipeline (OCR + chunking + embedding) for every PDF.</p>
  <form method="post">
    <label>Password</label>
    <input type="password" name="password" required>
    <input type="hidden" name="action" value="reprocess_all">
    <button type="submit" style="background:#dc3545; color:white; border:none; border-radius:4px;">Reprocess All</button>
  </form>
  %(message)s
</body>
</html>"""


def _html_response(status, message=""):
    body = HTML_FORM % {"message": message}
    return {
        "statusCode": status,
        "headers": {"Content-Type": "text/html; charset=utf-8"},
        "body": body,
    }


def _verify_password(password: str) -> bool:
    """Verify password against SHA-256 hash (simple, no bcrypt dependency needed)."""
    h = hashlib.sha256(password.encode()).hexdigest()
    return h == PASSWORD_HASH


def _parse_multipart(event):
    """Parse multipart/form-data from Lambda Function URL event."""
    content_type = event.get("headers", {}).get("content-type", "")
    if "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].strip()

    if event.get("isBase64Encoded"):
        body = base64.b64decode(event["body"])
    else:
        body = event["body"].encode()

    parts = body.split(f"--{boundary}".encode())
    fields = {}
    files = {}

    for part in parts:
        if b"Content-Disposition" not in part:
            continue

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        header = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4 :]
        if content.endswith(b"\r\n"):
            content = content[:-2]

        name = None
        filename = None
        for h in header.split("\r\n"):
            if "Content-Disposition" in h:
                for param in h.split(";"):
                    param = param.strip()
                    if param.startswith("name="):
                        name = param.split("=")[1].strip('"')
                    elif param.startswith("filename="):
                        filename = param.split("=")[1].strip('"')

        if name and filename:
            files[name] = {"filename": filename, "content": content}
        elif name:
            fields[name] = content.decode("utf-8", errors="replace")

    return fields, files


def _json_response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


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

    return _json_response(200, {"upload_url": presigned_url, "s3_key": s3_key})


def _parse_form_urlencoded(event):
    """Parse application/x-www-form-urlencoded body."""
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return dict(urllib.parse.parse_qsl(body))


def _handle_reprocess_all(password):
    """Copy every PDF in uploads/ to itself, re-triggering the full pipeline."""
    if not _verify_password(password):
        return _html_response(403, '<div class="msg err">Invalid password.</div>')

    paginator = s3.get_paginator("list_objects_v2")
    pdf_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="uploads/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    if not pdf_keys:
        return _html_response(200, '<div class="msg info">No PDFs found in uploads/.</div>')

    for i, key in enumerate(pdf_keys):
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

    return _html_response(
        200,
        f'<div class="msg ok">Reprocessing triggered for <strong>{len(pdf_keys)}</strong> PDFs. '
        f"This will take a while &mdash; each PDF goes through OCR, chunking, and embedding.</div>",
    )


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("requestContext", {}).get("http", {}).get("path", "/upload")

    # Presigned URL endpoint for large files
    if path.endswith("/presign") and method == "POST":
        return _handle_presign(event)

    if method == "GET":
        return _html_response(200)

    # POST: check content type to determine form type
    content_type = event.get("headers", {}).get("content-type", "")

    if "multipart/form-data" in content_type:
        # File upload form
        fields, files = _parse_multipart(event)

        # Check if this is a reprocess request (hidden field, no file)
        if fields.get("action") == "reprocess_all":
            return _handle_reprocess_all(fields.get("password", ""))

        password = fields.get("password", "")
        if not _verify_password(password):
            return _html_response(403, '<div class="msg err">Invalid password.</div>')

        file_data = files.get("file")
        if not file_data:
            return _html_response(400, '<div class="msg err">No file provided.</div>')

        filename = file_data["filename"]
        if not filename.lower().endswith(".pdf"):
            return _html_response(400, '<div class="msg err">Only PDF files are accepted.</div>')

        safe_name = os.path.basename(filename)
        s3_key = f"uploads/{safe_name}"

        s3.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=file_data["content"],
            ContentType="application/pdf",
        )

        escaped = html.escape(safe_name)
        return _html_response(
            200,
            f'<div class="msg ok">Uploaded <strong>{escaped}</strong> successfully. Processing will begin automatically.</div>',
        )

    # application/x-www-form-urlencoded (reprocess form)
    fields = _parse_form_urlencoded(event)
    if fields.get("action") == "reprocess_all":
        return _handle_reprocess_all(fields.get("password", ""))

    return _html_response(400, '<div class="msg err">Unknown request.</div>')
