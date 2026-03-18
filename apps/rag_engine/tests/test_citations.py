"""Tests for citation extraction, stripping, collapsing, and renumbering."""

from ras_rag_engine.citations import (
    collapse_duplicate_indices,
    extract_content,
    format_citations,
    make_index_map,
    renumber_response,
    renumber_text,
    strip_llm_sources,
)
from ras_rag_engine.retriever import RetrievedChunk


def _chunk(chunk_id: str = "c1", start_page: int = 1, end_page: int = 1, **kwargs) -> RetrievedChunk:
    defaults = dict(
        chunk_id=chunk_id,
        doc_id="doc1",
        text="some text",
        score=0.9,
        start_page=start_page,
        end_page=end_page,
        section_heading=None,
        source_filename="Author (2020) JMBRAS 90(1), 1-10.pdf",
        title="Test Article",
        author=None,
        year=None,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


# --- strip_llm_sources ---


class TestStripLlmSources:
    def test_strips_markdown_sources_at_end(self):
        text = "Some response text about history [1].\n\n---\n**Sources:**\n[1] Author (2020)"
        result = strip_llm_sources(text)
        assert result == "Some response text about history [1]."

    def test_strips_plain_source_at_end(self):
        text = "Response text here [1] [2].\n\nSource:\n[1] Foo\n[2] Bar"
        result = strip_llm_sources(text)
        assert result == "Response text here [1] [2]."

    def test_strips_references_at_end(self):
        text = "Long response text. " * 20 + "\n\n**References:**\n- Author (2020)\n- Author2 (2021)"
        result = strip_llm_sources(text)
        assert "References" not in result

    def test_preserves_mid_text_sources_mention(self):
        # "Sources" mentioned early in the text should not be stripped
        text = "The sources indicate that [1].\n\nFurther analysis shows [2]. This is the conclusion."
        result = strip_llm_sources(text)
        assert result == text

    def test_no_sources_section_returns_unchanged(self):
        text = "A response with [1] citations but no sources block."
        result = strip_llm_sources(text)
        assert result == text

    def test_strips_bibliography(self):
        text = "Long answer. " * 30 + "\n\n---\n**Bibliography:**\n1. Author (2020)"
        result = strip_llm_sources(text)
        assert "Bibliography" not in result


# --- collapse_duplicate_indices ---


class TestCollapseDuplicateIndices:
    def test_merges_same_chunk_refs(self):
        # Chunks 1 and 3 are the same chunk_id+start_page, chunk 2 is different
        chunks = [_chunk("c1", 1), _chunk("c2", 2), _chunk("c1", 1)]
        index_map = {1: 1, 2: 2, 3: 3}
        result = collapse_duplicate_indices(index_map, chunks)
        # [1] and [3] should both map to display 1, [2] maps to display 2
        assert result[1] == 1
        assert result[3] == 1
        assert result[2] == 2

    def test_renumbers_consecutively(self):
        chunks = [_chunk("c1", 1), _chunk("c1", 1), _chunk("c2", 2)]
        index_map = {1: 1, 2: 2, 3: 3}
        result = collapse_duplicate_indices(index_map, chunks)
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 2

    def test_no_duplicates_unchanged(self):
        chunks = [_chunk("c1", 1), _chunk("c2", 2), _chunk("c3", 3)]
        index_map = {1: 1, 2: 2, 3: 3}
        result = collapse_duplicate_indices(index_map, chunks)
        assert result == {1: 1, 2: 2, 3: 3}

    def test_empty_map(self):
        result = collapse_duplicate_indices({}, [])
        assert result == {}

    def test_preserves_appearance_order(self):
        # If chunk 5 appears first in text (display 1), then chunk 3 (display 2)
        chunks = [_chunk(f"c{i}", i) for i in range(1, 6)]
        index_map = {5: 1, 3: 2}  # chunk 5 first, then chunk 3
        result = collapse_duplicate_indices(index_map, chunks)
        assert result[5] == 1
        assert result[3] == 2


# --- renumber_response (end-to-end) ---


class TestRenumberResponse:
    def test_strips_llm_sources_and_appends_code_sources(self):
        chunks = [_chunk("c1", 1), _chunk("c2", 2)]
        text = "Fact one [1]. Fact two [2].\n\n---\n**Sources:**\n[1] Author\n[2] Author2"
        result = renumber_response(text, chunks)
        # Should have exactly one Sources section (code-generated)
        assert result.count("**Sources:**") == 1
        # The LLM sources should be stripped
        assert "[1] Author\n[2] Author2" not in result
        # Code-generated sources should be present
        assert "[1] Author (2020)" in result

    def test_collapses_duplicate_chunks(self):
        chunks = [_chunk("c1", 1), _chunk("c2", 2), _chunk("c1", 1)]
        text = "Claim A [1]. Claim B [2]. Claim C [3]."
        result = renumber_response(text, chunks)
        # [1] and [3] pointed to same chunk → should be same display number
        assert "Claim A [1]" in result
        assert "Claim B [2]" in result
        assert "Claim C [1]" in result  # [3] collapsed to [1]

    def test_think_block_preserved(self):
        chunks = [_chunk("c1", 1)]
        text = "<think>reasoning</think>Fact [1]."
        result = renumber_response(text, chunks)
        assert result.startswith("<think>reasoning</think>")
        assert "Fact [1]." in result

    def test_no_citations(self):
        chunks = [_chunk("c1", 1)]
        text = "A response with no citations."
        result = renumber_response(text, chunks)
        # Response with text but no [N] markers should show no sources
        assert "Sources" not in result

    def test_multi_citation_pattern(self):
        chunks = [_chunk("c1", 1), _chunk("c2", 2), _chunk("c3", 3)]
        text = "Fact [1, 3]. Other [2]."
        result = renumber_response(text, chunks)
        assert "[1, 2]" in result  # 1→1, 3→2 (renumbered)
        assert "Other [3]" in result  # 2→3


# --- strip + renumber integration ---


class TestStripAndRenumberIntegration:
    def test_response_with_llm_sources_gets_clean_output(self):
        chunks = [_chunk("c1", 1), _chunk("c2", 5, 7), _chunk("c3", 10)]
        text = (
            "Swettenham arrived in Singapore in 1874 [1]. He was appointed [2].\n"
            "Further details appear in [3].\n\n"
            "---\n**Source:**\n"
            "[1] Swettenham Journal\n"
            "[2] Abdullah (2011)\n"
            "[3] Aznan (2020)"
        )
        result = renumber_response(text, chunks)
        # Exactly one sources block
        assert result.count("**Sources:**") == 1
        # All 3 citations present in text
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        # Code-generated source lines
        assert "Author (2020)" in result
