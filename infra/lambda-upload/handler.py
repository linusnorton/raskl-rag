"""Upload Lambda: HTML form with password validation + S3 presigned URL upload."""

import base64
import hashlib
import html
import json
import os
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
  <title>raskl-rag Upload</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 480px; margin: 60px auto; padding: 0 20px; }
    h1 { font-size: 1.4em; }
    label { display: block; margin: 16px 0 4px; font-weight: 600; }
    input { width: 100%%; padding: 8px; box-sizing: border-box; }
    button { margin-top: 20px; padding: 10px 24px; cursor: pointer; }
    .msg { margin-top: 16px; padding: 12px; border-radius: 4px; }
    .ok { background: #d4edda; color: #155724; }
    .err { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h1>raskl-rag PDF Upload</h1>
  <form method="post" enctype="multipart/form-data">
    <label>Password</label>
    <input type="password" name="password" required>
    <label>PDF file</label>
    <input type="file" name="file" accept=".pdf" required>
    <button type="submit">Upload</button>
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


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")

    if method == "GET":
        return _html_response(200)

    # POST: parse form data
    fields, files = _parse_multipart(event)

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
