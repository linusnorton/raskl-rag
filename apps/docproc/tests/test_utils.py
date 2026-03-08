"""Unit tests for utility functions."""

from __future__ import annotations

from pathlib import Path

from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.geometry import bbox_contains, bbox_overlap, is_in_zone
from ras_docproc.utils.hashing import make_doc_id, slug, text_hash
from ras_docproc.utils.io import read_jsonl, write_jsonl
from ras_docproc.utils.text import clean_text, dehyphenate, normalize_nfkc, remove_script_intrusions, strip_diacritics


class TestSlug:
    def test_basic(self):
        assert slug("My Document.pdf") == "my-document"

    def test_special_chars(self):
        result = slug("Abdullah (2011) JMBRAS 84(1), 1-22.pdf")
        assert result == "abdullah-2011-jmbras-84-1-1-22"

    def test_unicode(self):
        result = slug("Ünïcödé File.pdf")
        assert result == "unicode-file"


class TestDocId:
    def test_deterministic(self):
        id1 = make_doc_id("test.pdf", "abc123def456")
        id2 = make_doc_id("test.pdf", "abc123def456")
        assert id1 == id2

    def test_format(self):
        result = make_doc_id("test.pdf", "abc123def456789")
        assert result == "test-abc123def456"


class TestTextHash:
    def test_deterministic(self):
        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        assert h1 == h2

    def test_nfkc_normalized(self):
        # Different Unicode representations should hash the same
        h1 = text_hash("\u00e9")  # é (precomposed)
        h2 = text_hash("\u0065\u0301")  # é (decomposed) - NFKC normalizes this
        assert h1 == h2


class TestBBoxOverlap:
    def test_no_overlap(self):
        a = BBox(x0=0, y0=0, x1=10, y1=10)
        b = BBox(x0=20, y0=20, x1=30, y1=30)
        assert bbox_overlap(a, b) == 0.0

    def test_full_overlap(self):
        a = BBox(x0=0, y0=0, x1=10, y1=10)
        assert bbox_overlap(a, a) == 1.0

    def test_partial_overlap(self):
        a = BBox(x0=0, y0=0, x1=10, y1=10)
        b = BBox(x0=5, y0=5, x1=15, y1=15)
        result = bbox_overlap(a, b)
        assert 0.0 < result < 1.0


class TestBBoxContains:
    def test_contains(self):
        outer = BBox(x0=0, y0=0, x1=100, y1=100)
        inner = BBox(x0=10, y0=10, x1=50, y1=50)
        assert bbox_contains(outer, inner) is True

    def test_not_contains(self):
        outer = BBox(x0=10, y0=10, x1=50, y1=50)
        inner = BBox(x0=0, y0=0, x1=100, y1=100)
        assert bbox_contains(outer, inner) is False


class TestIsInZone:
    def test_in_header_zone(self):
        bbox = BBox(x0=0, y0=10, x1=100, y1=50)
        assert is_in_zone(bbox, page_height=800, zone_top_frac=0.0, zone_bottom_frac=0.10) is True

    def test_not_in_header_zone(self):
        bbox = BBox(x0=0, y0=400, x1=100, y1=450)
        assert is_in_zone(bbox, page_height=800, zone_top_frac=0.0, zone_bottom_frac=0.10) is False

    def test_in_footer_zone(self):
        bbox = BBox(x0=0, y0=700, x1=100, y1=750)
        assert is_in_zone(bbox, page_height=800, zone_top_frac=0.85, zone_bottom_frac=1.0) is True


class TestNormalizeNFKC:
    def test_nfkc(self):
        result = normalize_nfkc("\ufb01")  # fi ligature
        assert result == "fi"


class TestDehyphenation:
    def test_dehyphenate(self):
        result = dehyphenate("con-\ntinued")
        assert result == "continued"

    def test_no_dehyphenate_uppercase(self):
        result = dehyphenate("end-\nStart")
        assert result == "end-\nStart"


class TestStripDiacritics:
    def test_strip_ocr_diacritics(self):
        assert strip_diacritics("Apríl") == "April"

    def test_strip_cafe(self):
        assert strip_diacritics("café") == "cafe"

    def test_preserves_base_ascii(self):
        assert strip_diacritics("hello") == "hello"

    def test_preserves_digits_punctuation(self):
        assert strip_diacritics("page 42, line 3.") == "page 42, line 3."

    def test_multiple_diacritics(self):
        assert strip_diacritics("naïve résumé") == "naive resume"


class TestRemoveScriptIntrusions:
    def test_remove_cyrillic(self):
        assert remove_script_intrusions("КЩаtext") == "text"

    def test_remove_cyrillic_mid_word(self):
        result = remove_script_intrusions("someКЩаtext")
        assert result == "sometext"

    def test_preserves_latin(self):
        assert remove_script_intrusions("hello world") == "hello world"

    def test_removes_cjk(self):
        result = remove_script_intrusions("text中文more")
        assert result == "textmore"

    def test_empty_string(self):
        assert remove_script_intrusions("") == ""


class TestCleanText:
    def test_soft_hyphen_removed(self):
        result = clean_text("soft\u00adhyphen")
        assert result == "softhyphen"

    def test_zero_width_removed(self):
        result = clean_text("zero\u200bwidth")
        assert result == "zerowidth"

    def test_whitespace_collapsed(self):
        result = clean_text("hello   world")
        assert result == "hello world"


class TestJsonlRoundtrip:
    def test_write_read(self, tmp_path: Path):
        records = [
            TextBlockRecord(
                block_id="test1",
                doc_id="doc1",
                page_num_1=1,
                bbox=BBox(x0=0, y0=0, x1=100, y1=100),
                text_raw="Hello world",
                text_clean="Hello world",
            ),
            TextBlockRecord(
                block_id="test2",
                doc_id="doc1",
                page_num_1=2,
                bbox=BBox(x0=10, y0=20, x1=200, y1=300),
                text_raw="Second block",
                text_clean="Second block",
            ),
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path, TextBlockRecord)
        assert len(loaded) == 2
        assert loaded[0].block_id == "test1"
        assert loaded[1].text_raw == "Second block"
        assert loaded[0].bbox.x0 == 0
