"""Backfill volume, issue, and page_range_label from JSONL into the documents table.

Reads each data/out/{doc_id}/documents.jsonl and updates the DB directly,
without re-embedding. Run init-db first to add the columns.

Usage:
    source .env
    uv run ras-chunker init-db          # adds columns if missing
    uv run python scripts/backfill_volume_issue.py
"""

import json
import os
from pathlib import Path

import psycopg


def main():
    dsn = os.environ.get("CHUNKER_DATABASE_DSN", "")
    if not dsn:
        print("Set CHUNKER_DATABASE_DSN env var (or source .env)")
        return

    data_dir = Path("data/out")
    if not data_dir.exists():
        print(f"No data directory at {data_dir}")
        return

    conn = psycopg.connect(dsn, autocommit=True)
    updated = 0
    skipped = 0

    for doc_dir in sorted(data_dir.iterdir()):
        docs_path = doc_dir / "documents.jsonl"
        if not docs_path.exists():
            continue

        raw = json.loads(docs_path.read_text().strip().splitlines()[0])

        # Merge overlay if present
        overlay_path = doc_dir / "documents_overlay.jsonl"
        if overlay_path.exists():
            overlay = json.loads(overlay_path.read_text().strip().splitlines()[0])
            raw.update(overlay)

        doc_id = raw["doc_id"]
        volume = raw.get("volume")
        issue = raw.get("issue")
        page_range_label = raw.get("page_range_label")

        result = conn.execute(
            """
            UPDATE documents
            SET volume = %(volume)s, issue = %(issue)s, page_range_label = %(page_range_label)s
            WHERE doc_id = %(doc_id)s
            """,
            {"doc_id": doc_id, "volume": volume, "issue": issue, "page_range_label": page_range_label},
        )
        if result.rowcount:
            updated += 1
            v = f"Vol.{volume}" if volume else "no vol"
            i = f"Part {issue}" if issue else "no issue"
            print(f"  Updated {doc_id}: {v}, {i}")
        else:
            skipped += 1

    conn.close()
    print(f"\nDone: {updated} updated, {skipped} not in DB")


if __name__ == "__main__":
    main()
