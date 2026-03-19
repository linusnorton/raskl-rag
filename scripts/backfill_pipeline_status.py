"""Backfill pipeline_status table from S3 status/*.json files.

Usage:
    # Source .env for DATABASE_DSN and AWS credentials
    source .env
    uv run python scripts/backfill_pipeline_status.py
"""

import json
import os

import boto3
import psycopg


def main():
    bucket = os.environ.get("DOCS_BUCKET", os.environ.get("ADMIN_S3_BUCKET", ""))
    dsn = os.environ.get("ADMIN_DATABASE_DSN", os.environ.get("CHUNKER_DATABASE_DSN", ""))

    if not bucket:
        print("Set DOCS_BUCKET or ADMIN_S3_BUCKET env var")
        return
    if not dsn:
        print("Set ADMIN_DATABASE_DSN or CHUNKER_DATABASE_DSN env var")
        return

    s3 = boto3.client("s3")
    conn = psycopg.connect(dsn, autocommit=True)

    paginator = s3.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix="status/"):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                status = json.loads(resp["Body"].read())
            except Exception as e:
                print(f"  Skip {obj['Key']}: {e}")
                continue

            filename = status.get("filename", "")
            stage = status.get("stage", "unknown")
            error = status.get("error")
            updated_at = status.get("updated_at")

            if not filename:
                continue

            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO pipeline_status (filename, stage, error, updated_at)
                       VALUES (%(filename)s, %(stage)s, %(error)s, %(updated_at)s)
                       ON CONFLICT (filename)
                       DO UPDATE SET stage = EXCLUDED.stage, error = EXCLUDED.error, updated_at = EXCLUDED.updated_at""",
                    {"filename": filename, "stage": stage, "error": error, "updated_at": updated_at},
                )
            count += 1
            if count % 100 == 0:
                print(f"  Backfilled {count} statuses...")

    conn.close()
    print(f"Done. Backfilled {count} pipeline statuses into DB.")


if __name__ == "__main__":
    main()
