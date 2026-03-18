"""Tests for metadata enrichment pipeline steps."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ras_docproc.schema import BBox, DocumentRecord, MetadataSource, TextBlockRecord


def _make_document(**kwargs) -> DocumentRecord:
    """Create a DocumentRecord with defaults."""
    defaults = {
        "doc_id": "test-doc-123456789012",
        "source_filename": "test.pdf",
        "sha256_pdf": "abc123",
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


def _make_block(page: int, text: str, block_type: str = "paragraph") -> TextBlockRecord:
    return TextBlockRecord(
        block_id=f"blk-{page}-{hash(text) % 10000}",
        doc_id="test-doc-123456789012",
        page_num_1=page,
        bbox=BBox(x0=0, y0=0, x1=100, y1=20),
        text_raw=text,
        text_clean=text,
        block_type=block_type,
    )


# --- Tests for extract_metadata provenance tracking ---


class TestExtractMetadataProvenance:
    """Verify provenance tracking in extract_metadata."""

    def test_pdf_metadata_tracked(self):
        from ras_docproc.pipeline.extract_metadata import extract_metadata

        doc = _make_document()
        blocks = {1: [_make_block(1, "Some text on page 1 that is long enough for title heuristic")]}
        pdf_meta = {"title": "A Good Title", "author": "John Smith", "page_count": "10"}

        doc = extract_metadata(doc, blocks, pdf_meta)

        assert doc.title == "A Good Title"
        assert doc.author == "John Smith"

        sources = {s.field: s.source for s in doc.metadata_sources}
        assert sources.get("title") == "pdf_metadata"
        assert sources.get("author") == "pdf_metadata"

    def test_cover_page_regex_tracked(self):
        from ras_docproc.pipeline.extract_metadata import extract_metadata

        doc = _make_document()
        cover_text = (
            "Source: Journal of the Malaysian Branch, Vol. 84, No. 1 (June 2011), pp. 1-22\n"
            "Author(s): Abdullah bin Abdul Kadir\n"
            "Stable URL: https://www.jstor.org/stable/12345\n"
            "DOI: 10.1234/jmbras.2011.001\n"
        )
        blocks = {1: [_make_block(1, cover_text)]}
        pdf_meta = {"page_count": "22"}

        doc = extract_metadata(doc, blocks, pdf_meta)

        assert doc.author == "Abdullah bin Abdul Kadir"
        assert doc.doi == "10.1234/jmbras.2011.001"
        assert doc.publication is not None
        assert doc.volume == "84"

        source_fields = {s.field for s in doc.metadata_sources}
        assert "author" in source_fields
        assert "doi" in source_fields
        assert "publication" in source_fields
        assert "volume" in source_fields

    def test_source_line_with_issue_number_before_year(self):
        """Year extracted from Source line where issue number precedes year in parens."""
        from ras_docproc.pipeline.extract_metadata import extract_metadata

        doc = _make_document()
        cover_text = (
            "Source: Journal of the Malaysian Branch of the Royal Asiatic Society, "
            "Vol. 47, No. 2 (226) (1974), pp. 1-82\n"
        )
        blocks = {1: [_make_block(1, cover_text)]}
        pdf_meta = {"page_count": "82"}

        doc = extract_metadata(doc, blocks, pdf_meta)

        assert doc.year == 1974

    def test_filename_pattern_tracked(self):
        from ras_docproc.pipeline.extract_metadata import extract_metadata

        doc = _make_document(source_filename="Smith (2015) Some Title.pdf")
        blocks = {1: [_make_block(1, "Just a short text block")]}
        pdf_meta = {"page_count": "10"}

        doc = extract_metadata(doc, blocks, pdf_meta)

        assert doc.author == "Smith"
        sources = {s.field: s for s in doc.metadata_sources}
        assert "author" in sources
        assert sources["author"].source == "filename_pattern"


# --- Tests for LLM metadata enrichment ---


class TestEnrichMetadataLLM:
    """Test the LLM-based metadata extraction."""

    def test_applies_llm_results_to_empty_fields(self):
        from ras_docproc.pipeline.enrich_metadata_llm import _apply_llm_result

        doc = _make_document()
        result = {
            "title": "The History of Malaya",
            "author": "Frank Swettenham",
            "editor": "R. O. Winstedt",
            "year": 1907,
            "abstract": "An account of British Malaya",
            "keywords": ["Malaya", "British colonialism", "Perak"],
            "language": "en",
            "description": "A primary source account of British administration in Malaya.",
            "isbn": "978-0-123456-78-9",
            "issn": "0126-7353",
            "series": "JMBRAS Monograph No. 5",
        }

        _apply_llm_result(doc, result)

        assert doc.title == "The History of Malaya"
        assert doc.author == "Frank Swettenham"
        assert doc.editor == "R. O. Winstedt"
        assert doc.year == 1907
        assert doc.abstract == "An account of British Malaya"
        assert doc.keywords == ["Malaya", "British colonialism", "Perak"]
        assert doc.language == "en"
        assert doc.description == "A primary source account of British administration in Malaya."
        assert doc.isbn == "978-0-123456-78-9"
        assert doc.issn == "0126-7353"
        assert doc.series == "JMBRAS Monograph No. 5"

        # Verify provenance
        llm_sources = [s for s in doc.metadata_sources if s.source == "llm_extraction"]
        assert len(llm_sources) >= 10  # all populated fields should be tracked

    def test_does_not_overwrite_existing_fields(self):
        from ras_docproc.pipeline.enrich_metadata_llm import _apply_llm_result

        doc = _make_document(title="Original Title", author="Original Author", year=2000)
        result = {
            "title": "LLM Title",
            "author": "LLM Author",
            "year": 1999,
            "abstract": "LLM abstract",  # this should be applied (was None)
        }

        _apply_llm_result(doc, result)

        assert doc.title == "Original Title"  # preserved
        assert doc.author == "Original Author"  # preserved
        assert doc.year == 2000  # preserved
        assert doc.abstract == "LLM abstract"  # filled

    def test_keywords_merge_not_replace(self):
        from ras_docproc.pipeline.enrich_metadata_llm import _apply_llm_result

        doc = _make_document()
        doc.keywords = ["existing_keyword"]
        result = {"keywords": ["existing_keyword", "new_keyword", "another"]}

        _apply_llm_result(doc, result)

        assert "existing_keyword" in doc.keywords
        assert "new_keyword" in doc.keywords
        assert "another" in doc.keywords
        assert doc.keywords.count("existing_keyword") == 1  # no duplication

    def test_invalid_year_skipped(self):
        from ras_docproc.pipeline.enrich_metadata_llm import _apply_llm_result

        doc = _make_document()
        result = {"year": "not_a_number"}

        _apply_llm_result(doc, result)
        assert doc.year is None

    @patch("boto3.client")
    def test_full_llm_enrichment_with_mock(self, mock_boto3_client):
        from ras_docproc.config import PipelineConfig
        from ras_docproc.pipeline.enrich_metadata_llm import enrich_metadata_llm

        # Mock Bedrock response
        llm_response = json.dumps(
            {
                "title": "Swettenham's Perak Journals",
                "author": "Frank A. Swettenham",
                "year": 1874,
                "language": "en",
                "keywords": ["Perak", "colonial administration"],
                "description": "Journals kept by Swettenham during his time in Perak.",
                "document_type": "primary_source",
            }
        )

        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.converse.return_value = {"output": {"message": {"content": [{"text": llm_response}]}}}

        config = PipelineConfig(pdf_path="/tmp/test.pdf")
        doc = _make_document()
        blocks = {
            1: [_make_block(1, "October 6th. Left Penang this morning for Perak. " * 5)],
            2: [_make_block(2, "The river was very full and the current strong. " * 5)],
        }

        doc = enrich_metadata_llm(doc, blocks, config)

        assert doc.title == "Swettenham's Perak Journals"
        assert doc.language == "en"
        assert "Perak" in doc.keywords
        mock_client.converse.assert_called_once()


# --- Tests for web metadata enrichment ---


class TestEnrichMetadataWeb:
    """Test the web API-based metadata enrichment."""

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_crossref_enrichment(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_crossref

        mock_get_json.return_value = {
            "message": {
                "title": ["The Malay Annals"],
                "author": [{"given": "C. C.", "family": "Brown"}],
                "editor": [{"given": "R. O.", "family": "Winstedt"}],
                "published-print": {"date-parts": [[1952]]},
                "container-title": ["Journal of the Malayan Branch of the Royal Asiatic Society"],
                "volume": "25",
                "issue": "2",
                "page": "5-276",
                "ISSN": ["0126-7353"],
                "subject": ["History", "Asian Studies"],
                "URL": "https://doi.org/10.1234/test",
                "abstract": "<p>A translation of the Sejarah Melayu.</p>",
            }
        }

        doc = _make_document(doi="10.1234/test")
        _enrich_from_crossref(doc)

        assert doc.title == "The Malay Annals"
        assert doc.author == "C. C. Brown"
        assert doc.editor == "R. O. Winstedt"
        assert doc.year == 1952
        assert doc.volume == "25"
        assert doc.issue == "2"
        assert doc.page_range_label == "5-276"
        assert doc.issn == "0126-7353"
        assert doc.abstract == "A translation of the Sejarah Melayu."  # JATS tags stripped
        assert "History" in doc.keywords
        assert doc.url == "https://doi.org/10.1234/test"

        # Verify provenance
        crossref_sources = [s for s in doc.metadata_sources if s.source == "crossref"]
        assert len(crossref_sources) >= 8

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_crossref_no_doi(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_crossref

        doc = _make_document()  # no DOI
        _enrich_from_crossref(doc)
        mock_get_json.assert_not_called()

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_crossref_does_not_overwrite(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_crossref

        mock_get_json.return_value = {
            "message": {
                "title": ["CrossRef Title"],
                "author": [{"given": "A", "family": "B"}],
                "published-print": {"date-parts": [[2000]]},
            }
        }

        doc = _make_document(doi="10.1234/test", title="Original", author="Original Author", year=1999)
        _enrich_from_crossref(doc)

        assert doc.title == "Original"
        assert doc.author == "Original Author"
        assert doc.year == 1999

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_openlibrary_enrichment(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_openlibrary

        mock_get_json.return_value = {
            "docs": [
                {
                    "title": "The Real Malay",
                    "author_name": ["Frank Swettenham"],
                    "isbn": ["978-0-123456-78-9"],
                    "publisher": ["John Lane"],
                    "first_publish_year": 1900,
                    "subject": ["Malay Peninsula", "British Malaya", "Colonial history"],
                }
            ]
        }

        doc = _make_document(title="The Real Malay")
        _enrich_from_openlibrary(doc)

        assert doc.isbn == "978-0-123456-78-9"
        assert doc.publication == "John Lane"
        assert doc.year == 1900
        assert "Malay Peninsula" in doc.keywords

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_openlibrary_no_title(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_openlibrary

        doc = _make_document()  # no title
        _enrich_from_openlibrary(doc)
        mock_get_json.assert_not_called()

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_duckduckgo_enrichment(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import _enrich_from_duckduckgo

        mock_get_json.return_value = {
            "AbstractText": "Frank Swettenham was a British colonial administrator in Malaya.",
            "AbstractSource": "Wikipedia",
            "RelatedTopics": [
                {"Text": "Perak War - A conflict in 1875-1876 between British and Malay forces."},
            ],
        }

        doc = _make_document(title="Swettenham Journals")
        _enrich_from_duckduckgo(doc)

        assert doc.description is not None
        assert "colonial administrator" in doc.description

        sources = {s.field: s.source for s in doc.metadata_sources}
        assert "description" in sources
        assert "duckduckgo" in sources["description"]

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_full_web_enrichment(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import enrich_metadata_web

        # CrossRef returns data, others return None
        def side_effect(url):
            if "crossref.org" in url:
                return {
                    "message": {
                        "title": ["Test Article"],
                        "author": [{"given": "J.", "family": "Smith"}],
                        "published-print": {"date-parts": [[2011]]},
                        "container-title": ["JMBRAS"],
                        "volume": "84",
                        "ISSN": ["0126-7353"],
                    }
                }
            return None

        mock_get_json.side_effect = side_effect

        doc = _make_document(doi="10.1234/test")
        doc = enrich_metadata_web(doc)

        assert doc.title == "Test Article"
        assert doc.author == "J. Smith"
        assert doc.year == 2011
        assert doc.issn == "0126-7353"

    @patch("ras_docproc.pipeline.enrich_metadata_web._http_get_json")
    def test_web_enrichment_handles_failures(self, mock_get_json):
        from ras_docproc.pipeline.enrich_metadata_web import enrich_metadata_web

        mock_get_json.side_effect = Exception("Network error")

        doc = _make_document(doi="10.1234/test", title="Test")
        doc = enrich_metadata_web(doc)

        # Should not crash, just log warnings
        assert doc.doc_id == "test-doc-123456789012"


# --- Tests for MetadataSource model ---


class TestMetadataSource:
    def test_metadata_source_serialization(self):
        source = MetadataSource(field="title", source="crossref", confidence=1.0, raw_value="Test Title")
        data = source.model_dump()
        assert data["field"] == "title"
        assert data["source"] == "crossref"
        assert data["confidence"] == 1.0

    def test_document_record_new_fields(self):
        doc = _make_document(
            abstract="An abstract",
            keywords=["kw1", "kw2"],
            language="en",
            isbn="978-1234",
            issn="0126-7353",
            series="JMBRAS Monograph No. 5",
            description="A description",
        )
        data = doc.model_dump()
        assert data["abstract"] == "An abstract"
        assert data["keywords"] == ["kw1", "kw2"]
        assert data["language"] == "en"
        assert data["isbn"] == "978-1234"
        assert data["issn"] == "0126-7353"
        assert data["series"] == "JMBRAS Monograph No. 5"
        assert data["description"] == "A description"
        assert data["metadata_sources"] == []

    def test_document_record_defaults(self):
        doc = _make_document()
        assert doc.abstract is None
        assert doc.keywords == []
        assert doc.language is None
        assert doc.isbn is None
        assert doc.issn is None
        assert doc.series is None
        assert doc.description is None
        assert doc.metadata_sources == []
