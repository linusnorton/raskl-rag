"""Tests for DeepSeek-OCR extraction backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from ras_docproc.config import PipelineConfig
from ras_docproc.pipeline.extract_deepseek import (
    _classify_block_type,
    _parse_grounded_markdown,
    _strip_markdown_markers,
    extract_with_deepseek,
)
from ras_docproc.schema import TextBlockRecord
from ras_docproc.utils.text import normalize_superscript_refs


class TestClassifyBlockType:
    """Test markdown-based block type classification."""

    def test_heading(self):
        assert _classify_block_type("## Introduction") == "heading"
        assert _classify_block_type("# Title") == "heading"
        assert _classify_block_type("### Sub-heading") == "heading"

    def test_paragraph(self):
        assert _classify_block_type("This is normal text.") == "paragraph"

    def test_list_item_dash(self):
        assert _classify_block_type("- first item") == "list_item"

    def test_list_item_star(self):
        assert _classify_block_type("* starred item") == "list_item"

    def test_list_item_numbered(self):
        assert _classify_block_type("1. first") == "list_item"
        assert _classify_block_type("2. second") == "list_item"

    def test_table(self):
        assert _classify_block_type("| col1 | col2 |") == "table"

    def test_table_without_trailing_pipe_is_paragraph(self):
        assert _classify_block_type("| partial row") == "paragraph"


class TestStripMarkdownMarkers:
    """Test stripping of markdown markers."""

    def test_heading_stripped(self):
        assert _strip_markdown_markers("## Introduction") == "Introduction"

    def test_list_stripped(self):
        assert _strip_markdown_markers("- item text") == "item text"

    def test_numbered_list_stripped(self):
        assert _strip_markdown_markers("1. first item") == "first item"

    def test_plain_text_unchanged(self):
        assert _strip_markdown_markers("plain text") == "plain text"


class TestGroundingParser:
    """Test _parse_grounded_markdown with synthetic grounding output."""

    DOC_ID = "test-doc-abc123"
    PAGE_WIDTH = 612.0
    PAGE_HEIGHT = 792.0
    PAGE_NUM = 1

    def _parse(self, text: str) -> list[TextBlockRecord]:
        return _parse_grounded_markdown(text, self.PAGE_WIDTH, self.PAGE_HEIGHT, self.DOC_ID, self.PAGE_NUM)

    def test_single_paragraph(self):
        response = "text[[100, 200, 500, 300]]\nHello world"
        blocks = self._parse(response)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.text_raw == "Hello world"
        assert b.block_type == "paragraph"
        assert b.doc_id == self.DOC_ID
        assert b.page_num_1 == self.PAGE_NUM
        assert b.reading_order == 0

    def test_title_tag_becomes_heading(self):
        response = "title[[0, 0, 999, 50]]\n# Introduction"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].block_type == "heading"
        assert blocks[0].text_raw == "Introduction"  # markdown markers stripped

    def test_sub_title_tag_becomes_heading(self):
        response = "sub_title[[100, 100, 800, 150]]\nChapter Overview"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].block_type == "heading"
        assert blocks[0].text_raw == "Chapter Overview"

    def test_text_tag_list_item(self):
        response = "text[[100, 100, 500, 150]]\n- First item"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].block_type == "list_item"
        assert blocks[0].text_raw == "First item"

    def test_text_tag_table(self):
        response = "text[[50, 300, 900, 400]]\n| Col A | Col B |"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].block_type == "table"

    def test_image_tag_skipped(self):
        response = "image[[50, 50, 500, 400]]\ntext[[100, 450, 800, 500]]\nSome text after image"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].text_raw == "Some text after image"

    def test_image_caption_parsed(self):
        response = "image_caption[[100, 400, 600, 430]]\nFig. 1. Map of Perak"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].block_type == "paragraph"
        assert blocks[0].text_raw == "Fig. 1. Map of Perak"

    def test_bbox_denormalization_full_page(self):
        """0,0,999,999 should map to approximately full page dimensions."""
        response = "text[[0, 0, 999, 999]]\nFull page text"
        blocks = self._parse(response)
        assert len(blocks) == 1
        bbox = blocks[0].bbox
        assert bbox.x0 == pytest.approx(0.0, abs=0.1)
        assert bbox.y0 == pytest.approx(0.0, abs=0.1)
        assert bbox.x1 == pytest.approx(self.PAGE_WIDTH, abs=1.0)
        assert bbox.y1 == pytest.approx(self.PAGE_HEIGHT, abs=1.0)

    def test_bbox_denormalization_partial(self):
        """Test coordinates scale proportionally."""
        response = "text[[0, 0, 499, 499]]\nPartial"
        blocks = self._parse(response)
        bbox = blocks[0].bbox
        assert bbox.x1 == pytest.approx(self.PAGE_WIDTH * 499 / 999, abs=1.0)
        assert bbox.y1 == pytest.approx(self.PAGE_HEIGHT * 499 / 999, abs=1.0)

    def test_empty_content_skipped(self):
        response = "text[[0, 0, 100, 100]]\n   \ntext[[200, 200, 500, 300]]\nReal content"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].text_raw == "Real content"

    def test_multiple_blocks_reading_order(self):
        response = (
            "title[[100, 50, 800, 100]]\n## Title\n"
            "text[[100, 120, 800, 300]]\nFirst paragraph text here.\n"
            "text[[100, 320, 800, 500]]\nSecond paragraph text here."
        )
        blocks = self._parse(response)
        assert len(blocks) == 3
        assert blocks[0].reading_order == 0
        assert blocks[1].reading_order == 1
        assert blocks[2].reading_order == 2
        assert blocks[0].block_type == "heading"
        assert blocks[1].block_type == "paragraph"
        assert blocks[2].block_type == "paragraph"

    def test_multi_line_content(self):
        response = "text[[100, 100, 800, 400]]\nFirst line of paragraph.\nSecond line continues here."
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert "First line" in blocks[0].text_raw
        assert "Second line" in blocks[0].text_raw

    def test_deterministic_block_ids(self):
        """Same input should produce same block IDs."""
        response = "text[[100, 200, 500, 300]]\nConsistent text"
        blocks1 = self._parse(response)
        blocks2 = self._parse(response)
        assert blocks1[0].block_id == blocks2[0].block_id

    def test_different_text_different_ids(self):
        """Different text should produce different block IDs."""
        r1 = "text[[100, 200, 500, 300]]\nText A"
        r2 = "text[[100, 200, 500, 300]]\nText B"
        b1 = self._parse(r1)
        b2 = self._parse(r2)
        assert b1[0].block_id != b2[0].block_id

    def test_no_tags(self):
        """Plain text with no tags returns empty list."""
        blocks = self._parse("This is just plain markdown with no tags.")
        assert len(blocks) == 0

    def test_coords_without_spaces(self):
        """Coordinates without spaces between numbers should also parse."""
        response = "text[[100,200,500,300]]\nNo spaces in coords"
        blocks = self._parse(response)
        assert len(blocks) == 1
        assert blocks[0].text_raw == "No spaces in coords"


class TestNormalizeSuperscriptRefs:
    """Test superscript footnote ref normalization."""

    def test_html_sup(self):
        assert normalize_superscript_refs("word<sup>17</sup> more") == "word(17) more"

    def test_latex_sup(self):
        assert normalize_superscript_refs(r"word\(^{91}\) more") == "word(91) more"

    def test_multiple_html(self):
        text = "first<sup>1</sup> and second<sup>2</sup>"
        assert normalize_superscript_refs(text) == "first(1) and second(2)"

    def test_no_match_unchanged(self):
        text = "plain text with no refs"
        assert normalize_superscript_refs(text) == text

    def test_mixed_formats(self):
        text = r"html<sup>5</sup> and latex\(^{10}\)"
        assert normalize_superscript_refs(text) == "html(5) and latex(10)"

    def test_letter_suffix(self):
        assert normalize_superscript_refs("word<sup>3a</sup> more") == "word(3a) more"
        assert normalize_superscript_refs(r"word\(^{3a}\) more") == "word(3a) more"


def _is_vllm_available(base_url: str = "http://localhost:8000") -> bool:
    """Check if vLLM server is running."""
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


@pytest.mark.slow
class TestDeepSeekE2E:
    """End-to-end tests requiring a running vLLM server with DeepSeek-OCR."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_vllm(self):
        if not _is_vllm_available():
            pytest.skip("vLLM server not available at localhost:8000")

    def test_extract_swettenham_page76(self, swettenham_pdf: Path):
        """Extract page 76 with DeepSeek backend, verify TextBlockRecords."""
        config = PipelineConfig(
            pdf_path=swettenham_pdf,
            page_range="76",
            extraction_backend="deepseek",
        )
        blocks_by_page = extract_with_deepseek(config, "test-swettenham")

        assert len(blocks_by_page) > 0, "Expected at least one page of blocks"
        assert 76 in blocks_by_page, "Expected page 76 in results"

        blocks = blocks_by_page[76]
        assert len(blocks) > 0, "Expected at least one block on page 76"

        for block in blocks:
            assert isinstance(block, TextBlockRecord)
            assert block.text_raw.strip(), "Block should have non-empty text"
            assert block.bbox.x1 > block.bbox.x0, "bbox width should be positive"
            assert block.bbox.y1 > block.bbox.y0, "bbox height should be positive"

    def test_full_pipeline_deepseek_backend(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Run full pipeline on page 76 with deepseek backend, verify JSONL export."""
        from ras_docproc.pipeline.boilerplate import detect_boilerplate
        from ras_docproc.pipeline.detect_captions import detect_captions
        from ras_docproc.pipeline.detect_content_area import detect_content_area
        from ras_docproc.pipeline.detect_figures import detect_figures
        from ras_docproc.pipeline.detect_footnotes import detect_footnotes
        from ras_docproc.pipeline.detect_language import detect_languages
        from ras_docproc.pipeline.detect_rotation import detect_rotation
        from ras_docproc.pipeline.export_jsonl import export_all
        from ras_docproc.pipeline.extract_metadata import extract_metadata
        from ras_docproc.pipeline.extract_mupdf import extract_pdf_metadata, extract_with_mupdf
        from ras_docproc.pipeline.inventory import run_inventory
        from ras_docproc.pipeline.link_footnote_refs import apply_ref_markup, link_footnote_refs
        from ras_docproc.pipeline.normalize_text import normalize_blocks, ocr_cleanup_blocks
        from ras_docproc.schema import PageRecord
        from ras_docproc.utils.hashing import page_content_hash, text_hash
        from ras_docproc.utils.io import read_jsonl

        config = PipelineConfig(
            pdf_path=swettenham_pdf,
            out_dir=tmp_out_dir,
            page_range="76",
            force=True,
            extraction_backend="deepseek",
        )

        document = run_inventory(config)
        doc_id = document.doc_id

        blocks_by_page = extract_with_deepseek(config, doc_id)

        mupdf_data = extract_with_mupdf(config)
        pdf_metadata = extract_pdf_metadata(config)

        page_records: list[PageRecord] = []
        page_heights: dict[int, float] = {}
        for page_num, pd in sorted(mupdf_data.items()):
            page_heights[page_num] = pd.height
            block_hashes = [text_hash(b.text_raw) for b in blocks_by_page.get(page_num, [])]
            p_hash = page_content_hash(block_hashes) if block_hashes else None
            page_records.append(
                PageRecord(
                    doc_id=doc_id,
                    page_index_0=page_num - 1,
                    page_num_1=page_num,
                    width=pd.width,
                    height=pd.height,
                    page_rotation=pd.rotation,
                    page_hash=p_hash,
                    text_char_count=sum(len(b.text_raw) for b in blocks_by_page.get(page_num, [])),
                    image_count=len(pd.images),
                )
            )

        for page_num in blocks_by_page:
            if page_num not in {pr.page_num_1 for pr in page_records}:
                page_heights[page_num] = 800.0
                page_records.append(
                    PageRecord(
                        doc_id=doc_id,
                        page_index_0=page_num - 1,
                        page_num_1=page_num,
                        width=612.0,
                        height=792.0,
                        text_char_count=sum(len(b.text_raw) for b in blocks_by_page.get(page_num, [])),
                    )
                )
        page_records.sort(key=lambda p: p.page_num_1)

        document = extract_metadata(document, blocks_by_page, pdf_metadata)
        blocks_by_page = normalize_blocks(blocks_by_page)
        blocks_by_page, removed_blocks = detect_boilerplate(blocks_by_page, page_heights, config)
        page_records = detect_content_area(blocks_by_page, page_records)
        blocks_by_page = detect_languages(blocks_by_page, config)
        blocks_by_page = ocr_cleanup_blocks(blocks_by_page)
        blocks_by_page, footnotes = detect_footnotes(blocks_by_page, page_heights, config, doc_id)
        for page_num in blocks_by_page:
            blocks_by_page[page_num] = [b for b in blocks_by_page[page_num] if b.block_type != "footnote"]
        footnote_refs = link_footnote_refs(blocks_by_page, footnotes, mupdf_data, doc_id)
        blocks_by_page = apply_ref_markup(blocks_by_page, footnote_refs)
        page_records, page_rotations = detect_rotation(mupdf_data, page_records)
        figures = detect_figures(mupdf_data, config, doc_id, page_rotations)
        figures, plates = detect_captions(figures, blocks_by_page, doc_id)

        doc_dir = export_all(
            out_dir=config.out_dir,
            doc_id=doc_id,
            document=document,
            pages=page_records,
            blocks_by_page=blocks_by_page,
            removed_blocks=removed_blocks,
            footnotes=footnotes,
            footnote_refs=footnote_refs,
            figures=figures,
            plates=plates,
        )

        assert (doc_dir / "documents.jsonl").exists()
        assert (doc_dir / "pages.jsonl").exists()
        assert (doc_dir / "text_blocks.jsonl").exists()

        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        assert len(blocks) > 0, "Expected text blocks in JSONL export"
