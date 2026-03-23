"""Backfill all document metadata from S3 JSONL into the documents table.

Reads the latest version of each processed/{doc_id}/documents.jsonl from S3
and updates the DB directly, without re-embedding. This fixes metadata that was
lost due to stale local files or early pipeline versions missing fields.

Usage:
    source .env
    uv run ras-chunker init-db          # adds columns if missing
    uv run python scripts/backfill_volume_issue.py
"""

import json
import os
import re

import boto3
import psycopg

# All metadata fields that can be backfilled from JSONL (excludes doc_id, sha256_pdf, s3_prefix, indexed_at)
_META_FIELDS = [
    "title", "author", "editor", "year", "publication", "document_type",
    "abstract", "keywords", "language", "isbn", "issn", "series", "description",
    "volume", "issue", "page_range_label", "page_offset",
]


def get_latest_version_key(s3, bucket: str, doc_id: str) -> str | None:
    """Find the latest version prefix for a doc_id and return the documents.jsonl key."""
    prefix = f"processed/{doc_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    if not prefixes:
        return None

    def version_num(p: str) -> int:
        m = re.search(r"/v(\d+)/$", p)
        return int(m.group(1)) if m else 0

    latest = max(prefixes, key=version_num)
    return f"{latest}documents.jsonl"


def main():
    bucket = os.environ.get("ADMIN_S3_BUCKET", "")
    if not bucket:
        print("Set ADMIN_S3_BUCKET env var (or source .env)")
        return

    dsn = os.environ.get("CHUNKER_DATABASE_DSN", os.environ.get("CHAT_DATABASE_DSN", ""))
    if not dsn:
        dsn = "postgresql://raskl:raskl@localhost:5432/raskl_rag"

    s3 = boto3.client("s3")
    conn = psycopg.connect(dsn, autocommit=True)

    # Get all doc_ids from the DB
    rows = conn.execute("SELECT doc_id FROM documents").fetchall()
    doc_ids = [r[0] for r in rows]
    print(f"Found {len(doc_ids)} documents in DB")

    updated = 0
    skipped = 0

    for doc_id in sorted(doc_ids):
        key = get_latest_version_key(s3, bucket, doc_id)
        if not key:
            print(f"  SKIP {doc_id}: no S3 data")
            skipped += 1
            continue

        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            raw = json.loads(obj["Body"].read().decode().strip().splitlines()[0])
        except Exception as e:
            print(f"  SKIP {doc_id}: {e}")
            skipped += 1
            continue

        # Check for overlay (outside version dir)
        overlay_prefix = f"processed/{doc_id}/documents_overlay.jsonl"
        try:
            overlay_obj = s3.get_object(Bucket=bucket, Key=overlay_prefix)
            overlay = json.loads(overlay_obj["Body"].read().decode().strip().splitlines()[0])
            raw.update(overlay)
        except s3.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        # Build SET clause and params for all metadata fields
        params = {"doc_id": doc_id}
        set_parts = []
        for field in _META_FIELDS:
            val = raw.get(field)
            # Ensure keywords is a list (default empty)
            if field == "keywords":
                val = val if isinstance(val, list) else []
            # Ensure page_offset is int
            if field == "page_offset":
                val = val if isinstance(val, int) else 0
            params[field] = val
            set_parts.append(f"{field} = %({field})s")

        conn.execute(
            f"UPDATE documents SET {', '.join(set_parts)} WHERE doc_id = %(doc_id)s",
            params,
        )
        updated += 1
        author = raw.get("author") or "no author"
        vol = f"Vol.{raw.get('volume')}" if raw.get("volume") else "no vol"
        dtype = raw.get("document_type") or "no type"
        print(f"  Updated {doc_id}: {author}, {vol}, {dtype}")

    conn.close()
    print(f"\nDone: {updated} updated, {skipped} skipped")


if __name__ == "__main__":
    main()
