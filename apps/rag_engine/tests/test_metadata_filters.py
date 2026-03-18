"""Tests for structured metadata filtering on search_documents and find_images tools."""

from __future__ import annotations

from ras_rag_engine.tools import get_tool_definitions, _SEARCH_TOOL, _FIND_IMAGES_TOOL
from ras_rag_engine.config import RAGConfig


# ---------------------------------------------------------------------------
# Tool definition tests (no DB needed)
# ---------------------------------------------------------------------------


class TestSearchToolDefinition:
    """Verify search_documents tool exposes the new filter parameters."""

    def _props(self) -> dict:
        return _SEARCH_TOOL["function"]["parameters"]["properties"]

    def test_has_year_from(self):
        assert "year_from" in self._props()
        assert self._props()["year_from"]["type"] == "integer"

    def test_has_year_to(self):
        assert "year_to" in self._props()
        assert self._props()["year_to"]["type"] == "integer"

    def test_has_language(self):
        assert "language" in self._props()
        assert self._props()["language"]["type"] == "string"

    def test_has_publication(self):
        assert "publication" in self._props()
        assert self._props()["publication"]["type"] == "string"

    def test_only_query_required(self):
        required = _SEARCH_TOOL["function"]["parameters"]["required"]
        assert required == ["query"]


class TestFindImagesToolDefinition:
    """Verify find_images tool exposes the new filter parameters."""

    def _props(self) -> dict:
        return _FIND_IMAGES_TOOL["function"]["parameters"]["properties"]

    def test_has_document_type(self):
        assert "document_type" in self._props()

    def test_has_year_from(self):
        assert "year_from" in self._props()
        assert self._props()["year_from"]["type"] == "integer"

    def test_has_year_to(self):
        assert "year_to" in self._props()
        assert self._props()["year_to"]["type"] == "integer"

    def test_no_language_or_publication(self):
        # find_images doesn't have language/publication filters
        assert "language" not in self._props()
        assert "publication" not in self._props()

    def test_only_query_required(self):
        required = _FIND_IMAGES_TOOL["function"]["parameters"]["required"]
        assert required == ["query"]


class TestGetToolDefinitions:
    """Verify get_tool_definitions returns tools with the new parameters."""

    def test_search_tool_has_filters(self):
        config = RAGConfig(web_search_enabled=False)
        tools = get_tool_definitions(config)
        search = next(t for t in tools if t["function"]["name"] == "search_documents")
        props = search["function"]["parameters"]["properties"]
        assert all(k in props for k in ("year_from", "year_to", "language", "publication"))

    def test_find_images_tool_has_filters(self):
        config = RAGConfig(web_search_enabled=False)
        tools = get_tool_definitions(config)
        images = next(t for t in tools if t["function"]["name"] == "find_images")
        props = images["function"]["parameters"]["properties"]
        assert all(k in props for k in ("document_type", "year_from", "year_to"))
