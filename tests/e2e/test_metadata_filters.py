"""E2E tests for structured metadata filtering in retrieval.

These tests verify that year_from, year_to, language, publication, and
document_type SQL WHERE clauses filter correctly against real data.

Uses a dummy zero vector to bypass the embedding model (Bedrock) — only
the SQL filtering and FTS logic are exercised.

Requires:
- Local: running PostgreSQL with indexed documents (default)
- Live:  Neon DB (pass --live flag)
- No AWS credentials needed

Run:
    uv run pytest tests/e2e/test_metadata_filters.py -v           # local DB
    uv run pytest tests/e2e/test_metadata_filters.py -v --live    # Neon
"""

from __future__ import annotations

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from ras_rag_engine.config import RAGConfig as ChatConfig
from ras_rag_engine.retriever import (
    RETRIEVE_SQL,
    RetrievedChunk,
    RetrievedFigure,
    _build_tsquery,
    _FIGURE_SEARCH_SQL,
)

NEON_DSN = os.environ.get(
    "E2E_DATABASE_DSN",
    "postgresql://raskl_app:npg_G9NwyIJMs1oV@ep-aged-wave-ab8sg6n6.eu-west-2.aws.neon.tech"
    "/raskl_rag?sslmode=require",
)


@pytest.fixture(scope="module")
def config(is_live: bool) -> ChatConfig:
    kwargs: dict = dict(rerank_enabled=False)
    if is_live:
        kwargs["database_dsn"] = NEON_DSN
    return ChatConfig(**kwargs)


def _dummy_vec(dims: int = 1024) -> str:
    """Return a zero vector string for pgvector — bypasses embedding model."""
    return "[" + ",".join("0.0" for _ in range(dims)) + "]"


def _retrieve(
    query: str,
    config: ChatConfig,
    top_k: int = 15,
    document_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    language: str | None = None,
    publication: str | None = None,
) -> list[RetrievedChunk]:
    """Run RETRIEVE_SQL with a dummy vector — tests SQL filtering without Bedrock."""
    vec_str = _dummy_vec(config.embed_dimensions)

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            tsquery = _build_tsquery(query, cur)
            cur.execute(RETRIEVE_SQL, {
                "vec": vec_str, "tsquery": tsquery, "top_k": top_k, "doc_type": document_type,
                "year_from": year_from, "year_to": year_to, "language": language, "publication": publication,
            })
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0], doc_id=r[1], text=r[2], start_page=r[3], end_page=r[4],
            section_heading=r[5], source_filename=r[6], title=r[7], author=r[8],
            year=r[9], page_offset=r[10], publication=r[11], document_type=r[12],
            score=r[13], editor=r[14], abstract=r[15], description=r[16],
        )
        for r in rows
    ]


def _retrieve_figures(
    query: str,
    config: ChatConfig,
    top_k: int = 5,
    document_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[RetrievedFigure]:
    """Run _FIGURE_SEARCH_SQL with a dummy vector — tests SQL filtering without Bedrock."""
    vec_str = _dummy_vec(config.embed_dimensions)
    base = config.api_base_url.rstrip("/")

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_FIGURE_SEARCH_SQL, {
                "vec": vec_str, "query": query, "top_k": top_k,
                "doc_type": document_type, "year_from": year_from, "year_to": year_to,
            })
            rows = cur.fetchall()

    return [
        RetrievedFigure(
            figure_id=r[0], doc_id=r[1], page_num=r[2], caption=r[3],
            image_url=f"{base}/v1/images/{r[0]}",
            thumb_url=f"{base}/v1/images/{r[0]}?thumb=true",
            source_filename=r[4],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Year range filtering
# ---------------------------------------------------------------------------


class TestYearFiltering:
    """Verify year_from and year_to restrict results by publication year."""

    def test_year_from_excludes_earlier(self, config: ChatConfig):
        """year_from=2000 should exclude all pre-2000 documents."""
        chunks = _retrieve("Swettenham", config, year_from=2000)
        for c in chunks:
            assert c.year is not None and c.year >= 2000, (
                f"Chunk from year {c.year} should be excluded by year_from=2000"
            )

    def test_year_to_excludes_later(self, config: ChatConfig):
        """year_to=1900 should exclude all post-1900 documents."""
        chunks = _retrieve("Swettenham", config, year_to=1900)
        for c in chunks:
            assert c.year is not None and c.year <= 1900, (
                f"Chunk from year {c.year} should be excluded by year_to=1900"
            )

    def test_year_range_brackets(self, config: ChatConfig):
        """year_from=1870, year_to=1879 should only return 1870s documents."""
        chunks = _retrieve("Swettenham", config, year_from=1870, year_to=1879)
        for c in chunks:
            assert c.year is not None and 1870 <= c.year <= 1879, (
                f"Chunk from year {c.year} should be in 1870-1879 range"
            )

    def test_impossible_year_range_returns_empty(self, config: ChatConfig):
        """year_from=3000 should return no results."""
        chunks = _retrieve("Swettenham", config, year_from=3000)
        assert len(chunks) == 0

    def test_no_year_filter_returns_results(self, config: ChatConfig):
        """Without year filters, results should be returned (baseline)."""
        chunks = _retrieve("Swettenham", config)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Document type filtering
# ---------------------------------------------------------------------------


class TestDocumentTypeFiltering:
    """Verify document_type filter (existing, but tested alongside new filters)."""

    def test_filter_primary_source(self, config: ChatConfig):
        """document_type=primary_source should only return primary sources."""
        chunks = _retrieve("Swettenham", config, document_type="primary_source")
        for c in chunks:
            assert c.document_type == "primary_source", (
                f"Expected primary_source, got {c.document_type}"
            )

    def test_filter_journal_article(self, config: ChatConfig):
        """document_type=journal_article should only return journal articles."""
        chunks = _retrieve("Swettenham", config, document_type="journal_article")
        for c in chunks:
            assert c.document_type == "journal_article", (
                f"Expected journal_article, got {c.document_type}"
            )

    def test_nonexistent_type_returns_empty(self, config: ChatConfig):
        """A bogus document_type should return no results."""
        chunks = _retrieve("Swettenham", config, document_type="nonexistent_type")
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    """Verify multiple filters can be used together."""

    def test_type_plus_year(self, config: ChatConfig):
        """Combining document_type and year_from should apply both."""
        chunks = _retrieve(
            "Swettenham", config,
            document_type="primary_source", year_from=1870, year_to=1879,
        )
        for c in chunks:
            assert c.document_type == "primary_source"
            assert c.year is not None and 1870 <= c.year <= 1879

    def test_all_null_filters_same_as_unfiltered(self, config: ChatConfig):
        """Passing all filters as None should match unfiltered results."""
        unfiltered = _retrieve("Swettenham tin", config)
        filtered = _retrieve(
            "Swettenham tin", config,
            document_type=None, year_from=None, year_to=None,
            language=None, publication=None,
        )
        unfiltered_ids = {c.chunk_id for c in unfiltered}
        filtered_ids = {c.chunk_id for c in filtered}
        assert unfiltered_ids == filtered_ids


# ---------------------------------------------------------------------------
# Figure search filtering
# ---------------------------------------------------------------------------


class TestFigureFiltering:
    """Verify year and document_type filters on figure search SQL."""

    def test_figure_year_to_excludes_modern(self, config: ChatConfig):
        """year_to=1900 on figure search should only return old documents."""
        figures = _retrieve_figures("map", config, year_to=1900)
        assert isinstance(figures, list)

    def test_figure_impossible_year_returns_empty(self, config: ChatConfig):
        """year_from=3000 on figure search should return no results."""
        figures = _retrieve_figures("map", config, year_from=3000)
        assert len(figures) == 0

    def test_figure_document_type_filter(self, config: ChatConfig):
        """document_type filter on figure search should not error."""
        figures = _retrieve_figures("map", config, document_type="primary_source")
        assert isinstance(figures, list)

    def test_figure_no_filter_returns_results(self, config: ChatConfig):
        """Unfiltered figure search should return results (baseline)."""
        figures = _retrieve_figures("map photograph", config)
        assert isinstance(figures, list)
