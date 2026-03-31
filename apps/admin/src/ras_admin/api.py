"""Admin app: FastAPI routes for document management, diffs, and upload."""

from __future__ import annotations

import asyncio
import difflib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load .env for local development (Lambda gets credentials from IAM role)
if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    from dotenv import load_dotenv

    load_dotenv()

from . import db, s3
from .auth import COOKIE_NAME, authenticate_with_open_webui, create_session_token, get_current_user, set_session_cookie
from .config import AdminConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


config = AdminConfig()
app = FastAPI(lifespan=lifespan)
app.state.config = config

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(_static_dir, "favicon.ico"))


def _user_or_redirect(request: Request) -> dict[str, Any] | None:
    """Return user dict or None (caller should redirect)."""
    return get_current_user(request)


# --- Login ---


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    user_info = authenticate_with_open_webui(config, email, password)
    if user_info is None:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    token = create_session_token(config, user_info)
    response = RedirectResponse(url="/", status_code=303)
    set_session_cookie(response, token)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


# --- Dashboard ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    conn = db.get_connection(config)
    try:
        stats = db.get_dashboard_stats(conn)

        # Pipeline status counts from DB
        pipeline_stages = ["uploaded", "processing", "processed", "indexing", "done", "error"]
        stage_counts = {stage: 0 for stage in pipeline_stages}
        stage_counts.update(db.get_pipeline_stage_counts(conn))
        statuses = db.get_all_pipeline_statuses(conn)
    finally:
        conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "stats": stats,
            "stage_counts": stage_counts,
            "statuses": sorted(statuses, key=lambda s: s.get("filename", "")),
        },
    )


# --- Pipeline Status ---


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_status_page(request: Request, stage: str = ""):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    conn = db.get_connection(config)
    try:
        all_statuses = db.get_all_pipeline_statuses(conn)
    finally:
        conn.close()

    if stage:
        statuses = [s for s in all_statuses if s.get("stage") == stage]
    else:
        statuses = all_statuses

    return templates.TemplateResponse(
        "pipeline.html",
        {"request": request, "user": user, "statuses": statuses, "stage": stage},
    )


# --- Documents List ---


@app.get("/documents", response_class=HTMLResponse)
async def documents_list(request: Request, q: str = ""):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    conn = db.get_connection(config)
    try:
        if q:
            docs = db.search_documents(conn, q)
        else:
            docs = db.get_all_documents(conn)
    finally:
        conn.close()

    return templates.TemplateResponse(
        "documents.html",
        {
            "request": request,
            "user": user,
            "documents": docs,
            "query": q,
        },
    )


# --- Document Detail ---


@app.get("/documents/{doc_id}", response_class=HTMLResponse)
async def document_detail(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    conn = db.get_connection(config)
    try:
        doc = db.get_document(conn, doc_id)
        if doc is None:
            return HTMLResponse("<h1>Document not found</h1>", status_code=404)
        changes = db.get_chunk_changes(conn, doc_id)
        chunks = db.get_chunks_for_doc(conn, doc_id)
        figures = db.get_figures_for_doc(conn, doc_id)
    finally:
        conn.close()

    # Get S3 versions, overlay, and base (extracted) metadata
    versions = []
    overlay = {}
    base_meta = {}
    if config.s3_bucket:
        s3_client = s3.get_client()
        versions = s3.list_versions(s3_client, config.s3_bucket, doc_id)
        for v in versions:
            v["meta"] = s3.get_version_meta(s3_client, config.s3_bucket, v["prefix"])
        overlay = s3.download_overlay(s3_client, config.s3_bucket, doc_id) or {}
        # Load base (extracted) metadata from latest documents.jsonl
        if versions:
            prefix = versions[-1]["prefix"]
            base_records = s3.download_jsonl_as_list(s3_client, config.s3_bucket, f"{prefix}documents.jsonl")
            if base_records:
                base_meta = base_records[0]
    else:
        import json as _json

        overlay_path = Path("data/out") / doc_id / "documents_overlay.jsonl"
        if overlay_path.exists():
            overlay = _json.loads(overlay_path.read_text().strip().splitlines()[0])
        base_path = Path("data/out") / doc_id / "documents.jsonl"
        if base_path.exists():
            base_meta = _json.loads(base_path.read_text().strip().splitlines()[0])

    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "user": user,
            "doc": doc,
            "changes": changes,
            "chunks": chunks,
            "figures": figures,
            "versions": versions,
            "overlay": overlay,
            "base_meta": base_meta,
        },
    )


# --- Metadata Overlay ---


@app.get("/api/documents/{doc_id}/overlay")
async def get_overlay(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if config.s3_bucket:
        s3_client = s3.get_client()
        overlay = s3.download_overlay(s3_client, config.s3_bucket, doc_id) or {}
    else:
        overlay_path = Path("data/out") / doc_id / "documents_overlay.jsonl"
        if overlay_path.exists():
            import json

            overlay = json.loads(overlay_path.read_text().strip().splitlines()[0])
        else:
            overlay = {}

    return JSONResponse(overlay)


@app.put("/api/documents/{doc_id}/overlay")
async def save_overlay(request: Request, doc_id: str, reindex: bool = False):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    # Filter to editable fields only — this is the complete new overlay (replaces old)
    overlay = {k: v for k, v in body.items() if k in db._EDITABLE_COLUMNS}

    # Save overlay to S3 or local filesystem (delete if empty)
    if config.s3_bucket:
        s3_client = s3.get_client()
        if overlay:
            s3.upload_overlay(s3_client, config.s3_bucket, doc_id, overlay)
        else:
            try:
                s3_client.delete_object(
                    Bucket=config.s3_bucket, Key=f"processed/{doc_id}/documents_overlay.jsonl"
                )
            except Exception:
                pass
    else:
        import json

        overlay_path = Path("data/out") / doc_id / "documents_overlay.jsonl"
        if overlay:
            overlay_path.write_text(json.dumps(overlay, ensure_ascii=False) + "\n")
        elif overlay_path.exists():
            overlay_path.unlink()

    # Update DB directly for immediate visibility — merge base + overlay
    if overlay:
        conn = db.get_connection(config)
        try:
            db.update_document_metadata(conn, doc_id, overlay)
        finally:
            conn.close()

    # Optionally trigger reindex
    if reindex and config.s3_bucket:
        s3_client = s3.get_client()
        versions = s3.list_versions(s3_client, config.s3_bucket, doc_id)
        if versions:
            latest = versions[-1]
            s3.trigger_reindex(s3_client, config.s3_bucket, doc_id, latest["version"])
            return JSONResponse({"ok": True, "message": "Metadata saved, reindexing triggered"})

    msg = "Metadata saved"
    if reindex and not config.s3_bucket:
        msg += f". Run: uv run ras-chunker index --doc-id {doc_id}"
    return JSONResponse({"ok": True, "message": msg})


# --- Document Actions (API endpoints) ---


@app.get("/api/documents/{doc_id}/download")
async def download_document(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    conn = db.get_connection(config)
    try:
        doc = db.get_document(conn, doc_id)
    finally:
        conn.close()

    if doc is None:
        return JSONResponse({"error": "Document not found"}, status_code=404)

    s3_client = s3.get_client()
    s3_key = f"uploads/{doc['source_filename']}"
    url = s3.presign_get(s3_client, config.s3_bucket, s3_key)
    return RedirectResponse(url=url, status_code=302)


@app.post("/api/documents/{doc_id}/reprocess")
async def reprocess_document(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    conn = db.get_connection(config)
    try:
        doc = db.get_document(conn, doc_id)
    finally:
        conn.close()

    if doc is None:
        return JSONResponse({"error": "Document not found"}, status_code=404)

    # Update pipeline status in DB
    conn = db.get_connection(config)
    try:
        db.upsert_pipeline_status(conn, doc["source_filename"], "uploaded")
    finally:
        conn.close()

    s3_client = s3.get_client()
    s3_key = f"uploads/{doc['source_filename']}"
    s3.trigger_reprocess(s3_client, config.s3_bucket, s3_key)
    return JSONResponse({"ok": True, "message": f"Reprocessing triggered for {doc['source_filename']}"})


@app.post("/api/documents/{doc_id}/reindex")
async def reindex_document(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    s3_client = s3.get_client()
    versions = s3.list_versions(s3_client, config.s3_bucket, doc_id)
    if not versions:
        return JSONResponse({"error": "No versions found"}, status_code=404)

    latest = versions[-1]
    s3.trigger_reindex(s3_client, config.s3_bucket, doc_id, latest["version"])
    return JSONResponse({"ok": True, "message": f"Reindexing triggered for v{latest['version']}"})


@app.delete("/api/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = db.get_connection(config)
    try:
        doc = db.get_document(conn, doc_id)
        if doc is None:
            return JSONResponse({"error": "Document not found"}, status_code=404)

        # Delete from S3
        if config.s3_bucket:
            s3_client = s3.get_client()
            s3.delete_doc_from_s3(s3_client, config.s3_bucket, doc_id, doc["source_filename"])

        # Delete from DB (CASCADE) + pipeline status
        db.delete_document(conn, doc_id)
        db.delete_pipeline_status(conn, doc["source_filename"])
    finally:
        conn.close()

    return JSONResponse({"ok": True, "message": "Document deleted"})


# --- Diff API ---


@app.get("/diff/{doc_id}", response_class=HTMLResponse)
async def diff_page(request: Request, doc_id: str, v1: int = 0, v2: int = 0):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    conn = db.get_connection(config)
    try:
        doc = db.get_document(conn, doc_id)
        if doc is None:
            return HTMLResponse("<h1>Document not found</h1>", status_code=404)
    finally:
        conn.close()

    versions = []
    if config.s3_bucket:
        s3_client = s3.get_client()
        versions = s3.list_versions(s3_client, config.s3_bucket, doc_id)

    return templates.TemplateResponse(
        "diff.html",
        {"request": request, "user": user, "doc": doc, "versions": versions, "v1": v1, "v2": v2},
    )


@app.get("/api/diff/{doc_id}")
async def get_diff(request: Request, doc_id: str, v1: int = 0, v2: int = 0):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    s3_client = s3.get_client()
    jsonl_files = [
        "documents.jsonl",
        "text_blocks.jsonl",
        "pages.jsonl",
        "footnotes.jsonl",
        "footnote_refs.jsonl",
        "figures.jsonl",
        "removed_blocks.jsonl",
        "plates.jsonl",
    ]

    file_diffs = []
    for filename in jsonl_files:
        old_text = s3.download_jsonl(s3_client, config.s3_bucket, f"processed/{doc_id}/v{v1}/{filename}")
        new_text = s3.download_jsonl(s3_client, config.s3_bucket, f"processed/{doc_id}/v{v2}/{filename}")
        old_text = old_text or ""
        new_text = new_text or ""
        if old_text == new_text:
            continue

        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"v{v1}/{filename}",
                tofile=f"v{v2}/{filename}",
                lineterm="",
            )
        )
        if diff:
            added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
            file_diffs.append({"filename": filename, "diff": diff, "added": added, "removed": removed})

    return JSONResponse({"doc_id": doc_id, "v1": v1, "v2": v2, "file_diffs": file_diffs})


# --- JSONL Viewer API ---


@app.get("/api/documents/{doc_id}/jsonl/{filename}")
async def get_jsonl_file(request: Request, doc_id: str, filename: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    s3_client = s3.get_client()

    # Overlay lives outside version directories
    if filename == "documents_overlay.jsonl":
        key = f"processed/{doc_id}/documents_overlay.jsonl"
        records = s3.download_jsonl_as_list(s3_client, config.s3_bucket, key)
        if records is None:
            return JSONResponse({"error": f"File {filename} not found"}, status_code=404)
        return JSONResponse(records)

    prefix = s3.get_latest_version_prefix(s3_client, config.s3_bucket, doc_id)
    if not prefix:
        return JSONResponse({"error": "No versions found"}, status_code=404)

    records = s3.download_jsonl_as_list(s3_client, config.s3_bucket, f"{prefix}{filename}")
    if records is None:
        return JSONResponse({"error": f"File {filename} not found"}, status_code=404)

    return JSONResponse(records)


# --- Image Proxy ---


@app.get("/api/images/{doc_id}/{path:path}")
async def get_image(request: Request, doc_id: str, path: str):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Try S3 first
    if config.s3_bucket:
        conn = db.get_connection(config)
        try:
            doc = db.get_document(conn, doc_id)
        finally:
            conn.close()
        if doc and doc.get("s3_prefix"):
            s3_client = s3.get_client()
            s3_key = f"{doc['s3_prefix']}/{path}"
            url = s3.presign_get(s3_client, config.s3_bucket, s3_key)
            return RedirectResponse(url=url, status_code=302)

    # Fallback: local file
    local_path = Path("data/out") / doc_id / path
    if local_path.exists():
        return FileResponse(local_path)

    return JSONResponse({"error": "Image not found"}, status_code=404)


# --- Upload ---


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "user": user,
        },
    )


@app.post("/api/upload/presign")
async def upload_presign(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    body = await request.json()
    filename = body.get("filename", "")
    if not filename or not filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "filename must end with .pdf"}, status_code=400)

    s3_client = s3.get_client()
    url = s3.presign_upload(s3_client, config.s3_bucket, filename)

    # Track upload status in DB
    conn = db.get_connection(config)
    try:
        db.upsert_pipeline_status(conn, os.path.basename(filename), "uploaded")
    finally:
        conn.close()

    return JSONResponse({"upload_url": url, "s3_key": f"uploads/{os.path.basename(filename)}"})


@app.get("/api/upload/status")
async def upload_status(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db.get_connection(config)
    try:
        statuses = db.get_all_pipeline_statuses(conn)
    finally:
        conn.close()

    # Serialize datetimes for JSON
    for st in statuses:
        if st.get("updated_at") and hasattr(st["updated_at"], "isoformat"):
            st["updated_at"] = st["updated_at"].isoformat()

    return JSONResponse(statuses)


@app.post("/api/upload/reprocess-all")
async def reprocess_all_docs(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    s3_client = s3.get_client()
    pdf_keys = s3.list_uploaded_pdfs(s3_client, config.s3_bucket)

    conn = db.get_connection(config)
    filenames = []
    try:
        for key in pdf_keys:
            filename = key.rsplit("/", 1)[-1]
            db.upsert_pipeline_status(conn, filename, "uploaded")
            filenames.append(filename)
            s3.trigger_reprocess(s3_client, config.s3_bucket, key)
    finally:
        conn.close()

    return JSONResponse({"filenames": filenames, "message": f"Reprocessing {len(filenames)} PDFs"})


# --- Bulk Actions ---


@app.post("/api/documents/bulk-delete")
async def bulk_delete(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    doc_ids = body.get("doc_ids", [])
    if not doc_ids:
        return JSONResponse({"error": "No documents specified"}, status_code=400)

    conn = db.get_connection(config)
    deleted = 0
    try:
        for doc_id in doc_ids:
            doc = db.get_document(conn, doc_id)
            if doc:
                if config.s3_bucket:
                    s3_client = s3.get_client()
                    s3.delete_doc_from_s3(s3_client, config.s3_bucket, doc_id, doc["source_filename"])
                db.delete_document(conn, doc_id)
                deleted += 1
    finally:
        conn.close()

    return JSONResponse({"ok": True, "deleted": deleted})


@app.post("/api/documents/bulk-reprocess")
async def bulk_reprocess(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    body = await request.json()
    doc_ids = body.get("doc_ids", [])
    if not doc_ids:
        return JSONResponse({"error": "No documents specified"}, status_code=400)

    conn = db.get_connection(config)
    s3_client = s3.get_client()
    triggered = 0
    try:
        for doc_id in doc_ids:
            doc = db.get_document(conn, doc_id)
            if doc:
                s3_key = f"uploads/{doc['source_filename']}"
                db.upsert_pipeline_status(conn, doc["source_filename"], "uploaded")
                s3.trigger_reprocess(s3_client, config.s3_bucket, s3_key)
                triggered += 1
    finally:
        conn.close()

    return JSONResponse({"ok": True, "triggered": triggered, "message": f"Reprocessing {triggered} document(s)"})


@app.post("/api/documents/reindex-all")
async def reindex_all_docs(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    conn = db.get_connection(config)
    try:
        docs = db.get_all_documents(conn)
    finally:
        conn.close()

    s3_client = s3.get_client()
    triggered = 0
    for i, doc in enumerate(docs):
        versions = s3.list_versions(s3_client, config.s3_bucket, doc["doc_id"])
        if versions:
            latest = versions[-1]
            s3.trigger_reindex(s3_client, config.s3_bucket, doc["doc_id"], latest["version"])
            triggered += 1
            # Throttle to avoid exhausting Neon connection slots
            await asyncio.sleep(0.5)
            if triggered % 20 == 0:
                await asyncio.sleep(10)

    return JSONResponse({"ok": True, "triggered": triggered, "message": f"Reindexing {triggered} document(s)"})


@app.post("/api/documents/bulk-reindex")
async def bulk_reindex(request: Request):
    user = _user_or_redirect(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.s3_bucket:
        return JSONResponse({"error": "S3 not configured"}, status_code=400)

    body = await request.json()
    doc_ids = body.get("doc_ids", [])
    if not doc_ids:
        return JSONResponse({"error": "No documents specified"}, status_code=400)

    s3_client = s3.get_client()
    triggered = 0
    for doc_id in doc_ids:
        versions = s3.list_versions(s3_client, config.s3_bucket, doc_id)
        if versions:
            latest = versions[-1]
            s3.trigger_reindex(s3_client, config.s3_bucket, doc_id, latest["version"])
            triggered += 1

    return JSONResponse({"ok": True, "triggered": triggered, "message": f"Reindexing {triggered} document(s)"})


def main():
    uvicorn.run("ras_admin.api:app", host="0.0.0.0", port=config.port, reload=True)


if __name__ == "__main__":
    main()
