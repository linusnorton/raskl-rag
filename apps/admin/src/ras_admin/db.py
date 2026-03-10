"""Database queries for admin dashboard (read-only except delete)."""

from __future__ import annotations

from typing import Any

import psycopg

from .config import AdminConfig


def get_connection(config: AdminConfig) -> psycopg.Connection:
    return psycopg.connect(config.database_dsn, sslmode="prefer")


def get_dashboard_stats(conn: psycopg.Connection) -> dict[str, Any]:
    """Get counts for dashboard."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM chunks")
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM figures")
        total_figures = cur.fetchone()[0]

    return {"total_docs": total_docs, "total_chunks": total_chunks, "total_figures": total_figures}


def get_recent_activity(conn: psycopg.Connection, limit: int = 10) -> list[dict[str, Any]]:
    """Get recent chunk_changes with document info."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cc.doc_id, d.source_filename, d.title, cc.version, cc.indexed_at,
                   cc.chunks_total, cc.chunks_added, cc.chunks_removed, cc.chunks_text_changed, cc.chunks_unchanged
            FROM chunk_changes cc
            LEFT JOIN documents d ON cc.doc_id = d.doc_id
            ORDER BY cc.indexed_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_all_documents(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Get all documents with chunk counts."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.*, count(c.chunk_id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            GROUP BY d.doc_id
            ORDER BY d.indexed_at DESC
            """
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_document(conn: psycopg.Connection, doc_id: str) -> dict[str, Any] | None:
    """Get a single document by doc_id."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        return dict(zip(cols, row))


def get_chunk_changes(conn: psycopg.Connection, doc_id: str) -> list[dict[str, Any]]:
    """Get version history for a document."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM chunk_changes
            WHERE doc_id = %s
            ORDER BY version DESC
            """,
            (doc_id,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_chunks_for_doc(conn: psycopg.Connection, doc_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Get first N chunks for a document."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, chunk_index, start_page, end_page, section_heading,
                   left(text, 300) as text_preview, token_count
            FROM chunks
            WHERE doc_id = %s
            ORDER BY chunk_index
            LIMIT %s
            """,
            (doc_id, limit),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_figures_for_doc(conn: psycopg.Connection, doc_id: str) -> list[dict[str, Any]]:
    """Get all figures for a document."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT figure_id, page_num, caption, asset_path, thumb_path
            FROM figures
            WHERE doc_id = %s
            ORDER BY page_num
            """,
            (doc_id,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def delete_document(conn: psycopg.Connection, doc_id: str) -> bool:
    """Delete a document (CASCADE deletes chunks, figures, chunk_changes)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        deleted = cur.rowcount > 0
    conn.commit()
    return deleted


def search_documents(conn: psycopg.Connection, query: str) -> list[dict[str, Any]]:
    """Search documents by filename or title."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.*, count(c.chunk_id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            WHERE d.source_filename ILIKE %s OR d.title ILIKE %s
            GROUP BY d.doc_id
            ORDER BY d.indexed_at DESC
            """,
            (f"%{query}%", f"%{query}%"),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


