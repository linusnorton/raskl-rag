"""End-to-end pipeline tests using actual PDFs."""

from __future__ import annotations

from pathlib import Path

from ras_docproc.config import PipelineConfig
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
from ras_docproc.pipeline.extract_qwen3vl import extract_with_qwen3vl
from ras_docproc.pipeline.inventory import run_inventory
from ras_docproc.pipeline.link_footnote_refs import apply_ref_markup, link_footnote_refs
from ras_docproc.pipeline.normalize_text import normalize_blocks, ocr_cleanup_blocks
from ras_docproc.schema import (
    DocumentRecord,
    FigureRecord,
    FootnoteRecord,
    FootnoteRefRecord,
    PageRecord,
    TextBlockRecord,
)
from ras_docproc.utils.hashing import page_content_hash, text_hash
from ras_docproc.utils.io import read_jsonl


def _run_full_pipeline(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int | None = None,
    page_range: str | None = None,
) -> tuple[Path, str]:
    """Run the full pipeline and return (doc_dir, doc_id)."""
    config = PipelineConfig(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=max_pages,
        page_range=page_range,
        force=True,
    )

    # 1. Inventory
    document = run_inventory(config)
    doc_id = document.doc_id

    # 2. Qwen3 VL extraction
    blocks_by_page, vl_figure_pages = extract_with_qwen3vl(config, doc_id)

    # 3. MuPDF extraction
    mupdf_data = extract_with_mupdf(config)
    pdf_metadata = extract_pdf_metadata(config)

    # Build page records
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

    # 4. Extract metadata
    document = extract_metadata(document, blocks_by_page, pdf_metadata)

    # 5. Normalize
    blocks_by_page = normalize_blocks(blocks_by_page)

    # 6. Boilerplate
    blocks_by_page, removed_blocks = detect_boilerplate(blocks_by_page, page_heights, config)

    # 7. Content area detection
    page_records = detect_content_area(blocks_by_page, page_records)

    # 8. Language
    blocks_by_page = detect_languages(blocks_by_page, config)

    # 9. OCR cleanup
    blocks_by_page = ocr_cleanup_blocks(blocks_by_page)

    # 10. Footnotes
    blocks_by_page, footnotes = detect_footnotes(blocks_by_page, page_heights, config, doc_id)
    for page_num in blocks_by_page:
        blocks_by_page[page_num] = [b for b in blocks_by_page[page_num] if b.block_type != "footnote"]

    # 11. Footnote refs
    footnote_refs = link_footnote_refs(blocks_by_page, footnotes, mupdf_data, doc_id)

    # 12. Footnote ref markup
    blocks_by_page = apply_ref_markup(blocks_by_page, footnote_refs)

    # 13. Rotation
    page_records, page_rotations = detect_rotation(mupdf_data, page_records)

    # 14. Figures
    figures = detect_figures(mupdf_data, config, doc_id, page_rotations, vl_figure_pages)

    # 15. Captions
    figures, plates = detect_captions(figures, blocks_by_page, doc_id)

    # 16. Export
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

    return doc_dir, doc_id


class TestSwettenhamPipeline:
    """Tests using the Swettenham Journal PDF."""

    def test_produces_jsonl_outputs(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Run pipeline on first 10 pages, verify JSONL files exist and are valid."""
        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=10)

        # Check files exist
        assert (doc_dir / "documents.jsonl").exists()
        assert (doc_dir / "pages.jsonl").exists()
        assert (doc_dir / "text_blocks.jsonl").exists()

        # Check non-empty
        docs = read_jsonl(doc_dir / "documents.jsonl", DocumentRecord)
        assert len(docs) == 1
        assert docs[0].doc_id == doc_id

        pages = read_jsonl(doc_dir / "pages.jsonl", PageRecord)
        assert len(pages) > 0
        assert len(pages) <= 10

        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        assert len(blocks) > 0

        # Verify doc_id is deterministic (slug + sha256)
        assert "-" in doc_id
        parts = doc_id.rsplit("-", 1)
        assert len(parts[1]) == 12  # sha256[:12]

    def test_boilerplate_detected(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Run pipeline on first 30 pages, verify boilerplate detection."""
        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=30)

        removed_path = doc_dir / "removed_blocks.jsonl"
        assert removed_path.exists()

        removed = read_jsonl(removed_path, TextBlockRecord)
        assert len(removed) > 0

        # Check that at least some blocks are header/footer type
        types = {b.block_type for b in removed}
        assert types & {"header", "footer"}, f"Expected header/footer blocks in removed, got: {types}"

    def test_figures_extracted(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Run pipeline on pages known to have images. Full-page scans should be filtered."""
        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=30)

        figures_path = doc_dir / "figures.jsonl"
        assert figures_path.exists()

        pages = read_jsonl(doc_dir / "pages.jsonl", PageRecord)
        figures = read_jsonl(figures_path, FigureRecord)
        # Scanned PDFs: after filtering, figures should be far fewer than pages
        # (each page has a full-page scan image that should be skipped)
        assert len(figures) < len(pages), (
            f"Expected fewer figures ({len(figures)}) than pages ({len(pages)}) after filtering full-page scans"
        )

        if figures:
            for fig in figures[:3]:
                if fig.asset_original_path:
                    assert Path(fig.asset_original_path).exists(), f"Missing: {fig.asset_original_path}"
                if fig.asset_jpg_path:
                    assert Path(fig.asset_jpg_path).exists(), f"Missing: {fig.asset_jpg_path}"

    def test_metadata_extracted(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Verify title, author, or year are populated on DocumentRecord."""
        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=10)

        docs = read_jsonl(doc_dir / "documents.jsonl", DocumentRecord)
        assert len(docs) == 1
        doc = docs[0]

        # At least some metadata should be populated from the JSTOR cover page
        metadata_fields = [doc.title, doc.author, doc.year, doc.url, doc.doi]
        populated = [f for f in metadata_fields if f is not None]
        assert len(populated) >= 1, f"Expected at least 1 metadata field populated, got none: {doc}"

    def test_content_area_computed(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Verify PageRecord.content_bbox is populated for pages with body text."""
        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=10)

        pages = read_jsonl(doc_dir / "pages.jsonl", PageRecord)
        pages_with_content_bbox = [p for p in pages if p.content_bbox is not None]
        # Most pages with text should have a content bbox
        assert len(pages_with_content_bbox) >= 1, "Expected at least 1 page with content_bbox"


class TestAznanPipeline:
    """Tests using the Aznan PDF."""

    def test_rotation_detection(self, aznan_pdf: Path, tmp_out_dir: Path):
        """Verify page 12 has vertical text and rotation suggestion."""
        doc_dir, doc_id = _run_full_pipeline(aznan_pdf, tmp_out_dir)

        pages = read_jsonl(doc_dir / "pages.jsonl", PageRecord)

        # Find page 12
        page_12 = None
        for p in pages:
            if p.page_num_1 == 12:
                page_12 = p
                break

        assert page_12 is not None, "Page 12 not found in output"
        assert page_12.has_vertical_text is True, (
            f"Expected vertical text on page 12, ratio={page_12.vertical_text_ratio}"
        )
        assert page_12.suggested_rotation_cw in (90, 270), (
            f"Expected rotation 90 or 270, got {page_12.suggested_rotation_cw}"
        )

        # Check for rendered clip figure
        figures = read_jsonl(doc_dir / "figures.jsonl", FigureRecord)
        rendered_figs = [f for f in figures if f.derived_from == "rendered_clip" and f.page_num_1 == 12]
        assert len(rendered_figs) > 0, "Expected rendered_clip figure for rotated page 12"
        assert rendered_figs[0].applied_rotation_cw != 0

    def test_caption_fig2(self, aznan_pdf: Path, tmp_out_dir: Path):
        """Find a figure with caption containing 'Fig. 2' or 'Fig.2'."""
        doc_dir, doc_id = _run_full_pipeline(aznan_pdf, tmp_out_dir)

        figures = read_jsonl(doc_dir / "figures.jsonl", FigureRecord)
        # At minimum, check that figures were extracted
        assert len(figures) > 0, "Expected at least some figures from Aznan PDF"


class TestAbdullahPipeline:
    """Tests using the Abdullah PDF."""

    def test_footnotes_detected_and_linked(self, abdullah_pdf: Path, tmp_out_dir: Path):
        """Verify footnotes are detected and linked."""
        doc_dir, doc_id = _run_full_pipeline(abdullah_pdf, tmp_out_dir)

        footnotes_path = doc_dir / "footnotes.jsonl"
        assert footnotes_path.exists()

        footnotes = read_jsonl(footnotes_path, FootnoteRecord)
        assert len(footnotes) > 0, "Expected footnotes in Abdullah PDF"

        # Check footnote numbers are positive
        for fn in footnotes:
            assert fn.footnote_number > 0

        # Check no footnote blocks leaked into text_blocks
        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        footnote_blocks = [b for b in blocks if b.block_type == "footnote"]
        assert len(footnote_blocks) == 0, (
            f"Found {len(footnote_blocks)} footnote blocks in text_blocks.jsonl — "
            "they should only appear in footnotes.jsonl"
        )

        # Check footnote refs
        refs_path = doc_dir / "footnote_refs.jsonl"
        assert refs_path.exists()

        refs = read_jsonl(refs_path, FootnoteRefRecord)
        # At least some refs should exist
        if refs:
            match_types = {r.match_type for r in refs}
            assert match_types, "Expected match_type on footnote refs"

    def test_footnote_ref_markup(self, abdullah_pdf: Path, tmp_out_dir: Path):
        """Verify at least one text_clean contains [ref:N] markup."""
        doc_dir, doc_id = _run_full_pipeline(abdullah_pdf, tmp_out_dir)

        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        blocks_with_refs = [b for b in blocks if "[ref:" in b.text_clean]
        refs = read_jsonl(doc_dir / "footnote_refs.jsonl", FootnoteRefRecord)
        markup_types = {
            "regex_superscript", "superscript_span", "regex_space_superscript",
            "regex_superscript_xpage", "superscript_span_xpage", "regex_space_superscript_xpage",
            "regex_period_superscript", "regex_period_superscript_xpage",
        }
        markupable_refs = [r for r in refs if r.match_type in markup_types]
        if markupable_refs:
            assert len(blocks_with_refs) > 0, "Expected [ref:N] markup in text_clean when superscript refs exist"


class TestFootnotePipeline:
    """Test footnote detection, splitting, classification and ref linking on pre-processed data.

    Runs the footnote pipeline stages on existing text_blocks.jsonl rather than
    re-extracting through Qwen3 VL, so no Bedrock access is needed.
    """

    def _run_footnote_pipeline(
        self, abdullah_data_dir: Path, abdullah_pdf: Path
    ) -> tuple[list[FootnoteRecord], list[FootnoteRefRecord], dict[int, list[TextBlockRecord]]]:
        """Load pre-processed blocks and run footnote detection + ref linking."""
        from ras_docproc.pipeline.extract_mupdf import extract_with_mupdf

        doc_id = abdullah_data_dir.name

        # Load pre-processed text blocks (includes footnote blocks with block_type still set)
        all_blocks = read_jsonl(abdullah_data_dir / "text_blocks.jsonl", TextBlockRecord)

        # Also load the original footnote blocks (they were removed from text_blocks during export)
        original_fns = read_jsonl(abdullah_data_dir / "footnotes.jsonl", FootnoteRecord)

        # Reconstruct blocks_by_page, re-adding footnote text as footnote-typed blocks
        blocks_by_page: dict[int, list[TextBlockRecord]] = {}
        for b in all_blocks:
            blocks_by_page.setdefault(b.page_num_1, []).append(b)
        for fn in original_fns:
            block = TextBlockRecord(
                block_id=fn.footnote_id,
                doc_id=doc_id,
                page_num_1=fn.page_num_1,
                bbox=fn.bbox,
                text_raw=fn.text_raw,
                text_clean=fn.text_clean or fn.text_raw,
                block_type="footnote",
            )
            blocks_by_page.setdefault(fn.page_num_1, []).append(block)

        # Get page heights from MuPDF (needs the PDF file)
        config = PipelineConfig(pdf_path=abdullah_pdf, out_dir=abdullah_data_dir.parent, force=True)
        mupdf_data = extract_with_mupdf(config)
        page_heights = {pn: pd.height for pn, pd in mupdf_data.items()}

        # Run footnote detection (with splitting)
        blocks_by_page, footnotes = detect_footnotes(blocks_by_page, page_heights, config, doc_id)

        # Remove footnote blocks from body text for ref linking
        body_blocks: dict[int, list[TextBlockRecord]] = {}
        for pg, blks in blocks_by_page.items():
            body_blocks[pg] = [b for b in blks if b.block_type != "footnote"]

        # Run ref linking (regex only, pass mupdf_data for superscript detection too)
        refs = link_footnote_refs(body_blocks, footnotes, mupdf_data, doc_id)

        # Apply ref markup
        body_blocks = apply_ref_markup(body_blocks, refs)

        return footnotes, refs, body_blocks

    def test_multi_footnote_splitting(self, abdullah_data_dir: Path, abdullah_pdf: Path):
        """Verify merged footnote blocks are split into individual footnotes.

        Qwen3 VL merges consecutive footnotes like "35 Weld to CO 273/105. 36 Smith (1990)"
        into a single block. The pipeline should split these into separate FootnoteRecords.
        """
        footnotes, _, _ = self._run_footnote_pipeline(abdullah_data_dir, abdullah_pdf)
        fn_numbers = {fn.footnote_number for fn in footnotes}

        # Page 8 has merged block "39 King George V's diary. 40 Winstedt (1992: 126)."
        assert 39 in fn_numbers, "Footnote 39 should be detected"
        assert 40 in fn_numbers, "Footnote 40 should be split from merged block with 39"

        # Page 9 has merged block "45 King George V's diary, 13 January 1882. 46"
        assert 45 in fn_numbers, "Footnote 45 should be detected"
        assert 46 in fn_numbers, "Footnote 46 should be split from merged block with 45"

        # Page 8 has merged block "35 Weld to Kimberley, CO 273/105. 36"
        assert 35 in fn_numbers, "Footnote 35 should be detected"
        assert 36 in fn_numbers, "Footnote 36 should be split from merged block with 35"

    def test_footnote_type_classification(self, abdullah_data_dir: Path, abdullah_pdf: Path):
        """Verify footnotes are classified as citation/explanatory/mixed."""
        footnotes, _, _ = self._run_footnote_pipeline(abdullah_data_dir, abdullah_pdf)
        assert len(footnotes) > 0

        # All footnotes should have a valid type
        for fn in footnotes:
            assert fn.footnote_type in ("citation", "explanatory", "mixed"), (
                f"Footnote {fn.footnote_number} has invalid type: {fn.footnote_type}"
            )

        # Citation footnotes should include diary/journal and archival refs
        citation_fns = [fn for fn in footnotes if fn.footnote_type == "citation"]
        assert len(citation_fns) >= 5, (
            f"Expected at least 5 citation footnotes, got {len(citation_fns)}"
        )

        # Check specific known citations
        fn_by_num = {fn.footnote_number: fn for fn in footnotes}
        if 35 in fn_by_num:
            assert fn_by_num[35].footnote_type == "citation", (
                "Footnote 35 ('Weld to Kimberley, CO 273/105') should be citation"
            )
        if 39 in fn_by_num:
            assert fn_by_num[39].footnote_type == "citation", (
                "Footnote 39 ('King George V's diary') should be citation"
            )

    def test_parenthesized_refs_detected(self, abdullah_data_dir: Path, abdullah_pdf: Path):
        """Verify footnote refs are detected for parenthesized numbers.

        Qwen3 VL renders Abdullah's footnote markers as "(N)" in body text.
        """
        _, refs, _ = self._run_footnote_pipeline(abdullah_data_dir, abdullah_pdf)
        assert len(refs) > 0, "Expected footnote refs to be detected"

        match_types = {r.match_type for r in refs}
        assert "regex_parens" in match_types, (
            f"Expected regex_parens refs, got match types: {match_types}"
        )

    def test_ref_coverage(self, abdullah_data_dir: Path, abdullah_pdf: Path):
        """Verify most footnotes have a corresponding ref in body text."""
        footnotes, refs, _ = self._run_footnote_pipeline(abdullah_data_dir, abdullah_pdf)

        if not footnotes:
            pytest.skip("No footnotes detected")

        ref_fn_numbers = {r.footnote_number for r in refs}
        fn_numbers = {fn.footnote_number for fn in footnotes}
        coverage = len(ref_fn_numbers & fn_numbers) / len(fn_numbers)
        assert coverage >= 0.5, (
            f"Expected >=50% ref coverage, got {coverage:.0%} "
            f"({len(ref_fn_numbers & fn_numbers)}/{len(fn_numbers)})"
        )

    def test_citation_footnotes_cover_primary_sources(self, abdullah_data_dir: Path, abdullah_pdf: Path):
        """Verify diary/journal/dispatch footnotes are classified as citations.

        These are the references the RAG system needs to 'unpack' — when a
        journal article cites a colonial office dispatch or royal diary, the
        [cites:] marker should signal the LLM to mention the original source.
        """
        footnotes, _, _ = self._run_footnote_pipeline(abdullah_data_dir, abdullah_pdf)
        citation_fns = [fn for fn in footnotes if fn.footnote_type in ("citation", "mixed")]
        citation_texts = " ".join(fn.text_clean for fn in citation_fns)

        # These primary source types should be captured as citations
        assert "diary" in citation_texts.lower(), "Expected diary references in citation footnotes"
        assert "CO " in citation_texts or "CO\xa0" in citation_texts, (
            "Expected Colonial Office (CO) refs in citation footnotes"
        )


class TestOCRCleanup:
    """Test OCR diacritic cleanup on English blocks."""

    def test_no_combining_marks_in_english(self, swettenham_pdf: Path, tmp_out_dir: Path):
        """Verify English blocks don't contain unexpected combining marks after cleanup."""
        import unicodedata

        doc_dir, doc_id = _run_full_pipeline(swettenham_pdf, tmp_out_dir, max_pages=10)

        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        en_blocks = [b for b in blocks if b.lang == "en"]

        for block in en_blocks:
            for ch in block.text_clean:
                assert unicodedata.category(ch) != "Mn", (
                    f"English block {block.block_id} still has combining mark "
                    f"U+{ord(ch):04X} in text_clean: ...{block.text_clean[:80]}..."
                )


class TestLanguageDetection:
    """Test language detection across any PDF."""

    def test_language_fields_populated(self, abdullah_pdf: Path, tmp_out_dir: Path):
        """Verify text blocks have lang field populated."""
        doc_dir, doc_id = _run_full_pipeline(abdullah_pdf, tmp_out_dir, max_pages=5)

        blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
        assert len(blocks) > 0

        # Check that lang is set
        langs = [b.lang for b in blocks if b.lang is not None]
        assert len(langs) > 0, "Expected some blocks with lang set"

        # Check English detection
        en_blocks = [b for b in blocks if b.lang == "en" and b.lang_confidence is not None and b.lang_confidence > 0.5]
        assert len(en_blocks) > 0, "Expected English text blocks with confidence > 0.5"

        # Check short blocks
        short_blocks = [b for b in blocks if len(b.text_clean or b.text_raw) < 30]
        for sb in short_blocks:
            assert sb.lang == "unknown", f"Short block should be 'unknown', got '{sb.lang}'"


class TestDeterministicIds:
    """Test that IDs are deterministic across runs."""

    def test_deterministic(self, abdullah_pdf: Path, tmp_out_dir: Path):
        """Run pipeline twice, verify IDs match."""
        out1 = tmp_out_dir / "run1"
        out1.mkdir()
        out2 = tmp_out_dir / "run2"
        out2.mkdir()

        doc_dir1, doc_id1 = _run_full_pipeline(abdullah_pdf, out1, max_pages=3)
        doc_dir2, doc_id2 = _run_full_pipeline(abdullah_pdf, out2, max_pages=3)

        assert doc_id1 == doc_id2

        blocks1 = read_jsonl(doc_dir1 / "text_blocks.jsonl", TextBlockRecord)
        blocks2 = read_jsonl(doc_dir2 / "text_blocks.jsonl", TextBlockRecord)

        assert len(blocks1) == len(blocks2)
        ids1 = sorted(b.block_id for b in blocks1)
        ids2 = sorted(b.block_id for b in blocks2)
        assert ids1 == ids2, "Block IDs should be identical across runs"
