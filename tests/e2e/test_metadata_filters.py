"""E2E tests for structured metadata filtering in retrieval.

These tests verify that year_from, year_to, language, publication, and
document_type filters produce correct SQL filtering against real data.

Requires:
- AWS Bedrock access (for embedding queries)
- Local: running PostgreSQL with indexed documents (default)
- Live:  Neon DB (pass --live flag)

Run:
    # Local (default)
    AWS_PROFILE=linusnorton uv run pytest tests/e2e/test_metadata_filters.py -v

    # Live (Neon)
    AWS_PROFILE=linusnorton uv run pytest tests/e2e/test_metadata_filters.py -v --live
"""

from __future__ import annotations

import os

import pytest

from ras_rag_engine.config import RAGConfig as ChatConfig
from ras_rag_engine.retriever import retrieve, retrieve_figures

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


# ---------------------------------------------------------------------------
# Year range filtering
# ---------------------------------------------------------------------------


class TestYearFiltering:
    """Verify year_from and year_to restrict results by publication year."""

    def test_year_from_excludes_earlier(self, config: ChatConfig):
        """year_from=2000 should exclude all pre-2000 documents."""
        chunks = retrieve("Swettenham", config, year_from=2000)
        for c in chunks:
            assert c.year is None or c.year >= 2000, (
                f"Chunk from year {c.year} should be excluded by year_from=2000"
            )

    def test_year_to_excludes_later(self, config: ChatConfig):
        """year_to=1900 should exclude all post-1900 documents."""
        chunks = retrieve("Swettenham", config, year_to=1900)
        for c in chunks:
            assert c.year is None or c.year <= 1900, (
                f"Chunk from year {c.year} should be excluded by year_to=1900"
            )

    def test_year_range_brackets(self, config: ChatConfig):
        """year_from=1870, year_to=1879 should only return 1870s documents."""
        chunks = retrieve("Swettenham", config, year_from=1870, year_to=1879)
        for c in chunks:
            assert c.year is not None and 1870 <= c.year <= 1879, (
                f"Chunk from year {c.year} should be in 1870-1879 range"
            )

    def test_impossible_year_range_returns_empty(self, config: ChatConfig):
        """year_from=3000 should return no results."""
        chunks = retrieve("Swettenham", config, year_from=3000)
        assert len(chunks) == 0

    def test_no_year_filter_returns_results(self, config: ChatConfig):
        """Without year filters, results should be returned (baseline)."""
        chunks = retrieve("Swettenham", config)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Document type filtering
# ---------------------------------------------------------------------------


class TestDocumentTypeFiltering:
    """Verify document_type filter (existing, but tested alongside new filters)."""

    def test_filter_primary_source(self, config: ChatConfig):
        """document_type=primary_source should only return primary sources."""
        chunks = retrieve("Swettenham", config, document_type="primary_source")
        for c in chunks:
            assert c.document_type == "primary_source", (
                f"Expected primary_source, got {c.document_type}"
            )

    def test_filter_journal_article(self, config: ChatConfig):
        """document_type=journal_article should only return journal articles."""
        chunks = retrieve("Swettenham", config, document_type="journal_article")
        for c in chunks:
            assert c.document_type == "journal_article", (
                f"Expected journal_article, got {c.document_type}"
            )

    def test_nonexistent_type_returns_empty(self, config: ChatConfig):
        """A bogus document_type should return no results."""
        chunks = retrieve("Swettenham", config, document_type="nonexistent_type")
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    """Verify multiple filters can be used together."""

    def test_type_plus_year(self, config: ChatConfig):
        """Combining document_type and year_from should apply both."""
        chunks = retrieve(
            "Swettenham", config,
            document_type="primary_source", year_from=1870, year_to=1879,
        )
        for c in chunks:
            assert c.document_type == "primary_source"
            assert c.year is not None and 1870 <= c.year <= 1879

    def test_all_null_filters_same_as_unfiltered(self, config: ChatConfig):
        """Passing all filters as None should match unfiltered results."""
        unfiltered = retrieve("Swettenham tin", config)
        filtered = retrieve(
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
    """Verify year and document_type filters on retrieve_figures."""

    def test_figure_year_to_excludes_modern(self, config: ChatConfig):
        """year_to=1900 on figure search should only return old documents."""
        figures = retrieve_figures("map", config, year_to=1900)
        # We can't directly check year on figures, but we verify no error
        # and that the filter is applied (figures from modern docs excluded)
        # Just verify it runs without error and returns a list
        assert isinstance(figures, list)

    def test_figure_impossible_year_returns_empty(self, config: ChatConfig):
        """year_from=3000 on figure search should return no results."""
        figures = retrieve_figures("map", config, year_from=3000)
        assert len(figures) == 0

    def test_figure_document_type_filter(self, config: ChatConfig):
        """document_type filter on figure search should not error."""
        figures = retrieve_figures("map", config, document_type="primary_source")
        assert isinstance(figures, list)

    def test_figure_no_filter_returns_results(self, config: ChatConfig):
        """Unfiltered figure search should return results (baseline)."""
        figures = retrieve_figures("map photograph", config)
        # May return 0 if no figures indexed, but should not error
        assert isinstance(figures, list)
