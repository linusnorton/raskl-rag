"""Tests for the browse_corpus tool."""

from __future__ import annotations

import os

import pytest

from ras_rag_engine.config import RAGConfig
from ras_rag_engine.tools import (
    _BROWSE_CORPUS_TOOL,
    _execute_browse_corpus,
    _resolve_publication_pattern,
    get_tool_definitions,
)


# ---------------------------------------------------------------------------
# Tool definition tests (no DB needed)
# ---------------------------------------------------------------------------


class TestBrowseCorpusToolDefinition:
    """Verify browse_corpus tool definition is correct."""

    def _props(self) -> dict:
        return _BROWSE_CORPUS_TOOL["function"]["parameters"]["properties"]

    def test_has_action(self):
        assert "action" in self._props()
        assert self._props()["action"]["enum"] == ["list", "count", "overview"]

    def test_has_publication(self):
        assert "publication" in self._props()

    def test_has_volume(self):
        assert "volume" in self._props()

    def test_has_issue(self):
        assert "issue" in self._props()

    def test_has_author(self):
        assert "author" in self._props()

    def test_has_group_by(self):
        assert "group_by" in self._props()
        assert "author" in self._props()["group_by"]["enum"]

    def test_only_action_required(self):
        required = _BROWSE_CORPUS_TOOL["function"]["parameters"]["required"]
        assert required == ["action"]

    def test_registered_in_tool_definitions(self):
        config = RAGConfig(web_search_enabled=False)
        tools = get_tool_definitions(config)
        names = [t["function"]["name"] for t in tools]
        assert "browse_corpus" in names


class TestPublicationAliases:
    """Verify publication name normalization."""

    def test_jmbras_resolves(self):
        assert _resolve_publication_pattern("JMBRAS") == "%Royal Asiatic%"

    def test_jsbras_resolves(self):
        assert _resolve_publication_pattern("JSBRAS") == "%Straits Branch%"

    def test_case_insensitive(self):
        assert _resolve_publication_pattern("jmbras") == "%Royal Asiatic%"

    def test_unknown_uses_ilike(self):
        assert _resolve_publication_pattern("Nature") == "%Nature%"

    def test_none_returns_none(self):
        assert _resolve_publication_pattern(None) is None


# ---------------------------------------------------------------------------
# E2E tests against real database (requires CHAT_DATABASE_DSN)
# ---------------------------------------------------------------------------

def _get_config() -> RAGConfig:
    dsn = os.environ.get("CHAT_DATABASE_DSN", "")
    if not dsn:
        pytest.skip("CHAT_DATABASE_DSN not set")
    return RAGConfig(database_dsn=dsn)


class TestBrowseCorpusE2E:
    """E2E tests for browse_corpus against Neon database."""

    def test_list_jmbras_volume_87_part_1(self):
        """The query that started it all — list contents of JMBRAS Vol 87 Part 1."""
        config = _get_config()
        result, chunks = _execute_browse_corpus(
            {"action": "list", "publication": "JMBRAS", "volume": "87", "issue": "1"},
            config,
        )
        assert chunks == []  # browse_corpus returns no chunks

        # Should find the known articles in Vol 87 Part 1
        assert "Found" in result
        # Known articles (by author surname from filenames)
        assert "Hardwick" in result or "Horsing Around" in result
        assert "Taylor" in result or "Orientalist" in result
        assert "Hamzah" in result or "Non-fulfilment" in result

    def test_list_returns_author_from_db(self):
        """Documents should have author populated (not just filename fallback)."""
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "list", "publication": "JMBRAS", "volume": "87", "issue": "1"},
            config,
        )
        # At least some entries should show an author (not just title)
        # Taylor is known to have author in DB
        assert "Taylor" in result

    def test_count_by_publication(self):
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "count", "group_by": "publication"},
            config,
        )
        assert "Document count by publication" in result
        assert "Royal Asiatic" in result or "JMBRAS" in result

    def test_count_by_volume_for_jmbras(self):
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "count", "publication": "JMBRAS", "group_by": "volume"},
            config,
        )
        assert "Document count by volume" in result
        assert "87" in result

    def test_overview(self):
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "overview"},
            config,
        )
        assert "Corpus overview" in result
        assert "Total documents" in result

    def test_filter_by_author(self):
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "list", "author": "Gullick"},
            config,
        )
        assert "Found" in result
        assert "Gullick" in result

    def test_no_results_returns_message(self):
        config = _get_config()
        result, _ = _execute_browse_corpus(
            {"action": "list", "publication": "JMBRAS", "volume": "999"},
            config,
        )
        assert "No documents found" in result
