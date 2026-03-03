"""Compare DeepSeek vision OCR against existing pipeline OCR for scanned pages."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import fitz
import httpx
import pytest

from ras_docproc.config import PipelineConfig
from ras_docproc.pipeline.boilerplate import detect_boilerplate
from ras_docproc.pipeline.detect_footnotes import detect_footnotes
from ras_docproc.pipeline.detect_language import detect_languages
from ras_docproc.pipeline.extract_docling import extract_with_docling
from ras_docproc.pipeline.extract_mupdf import extract_with_mupdf
from ras_docproc.pipeline.inventory import run_inventory
from ras_docproc.pipeline.normalize_text import normalize_blocks, ocr_cleanup_blocks
from ras_docproc.schema import TextBlockRecord

VLLM_BASE_URL = os.environ.get("DOCPROC_VLLM_URL", "http://localhost:8000")
DEEPSEEK_MODEL = os.environ.get("DOCPROC_VLLM_MODEL", "deepseek-ai/DeepSeek-OCR")

# Known OCR errors on Swettenham page 76 — (garbled, correct)
KNOWN_ERRORS = [
    ("nt ", "at "),
    ("a,nd", "and"),
    ("ve fine", "very fine"),
    ("som wood", "some wood"),
    ("mak Penang", "make Penang"),
    ("Singapo", "Singapore"),
    ("Governo ", "Governor "),
]


@pytest.fixture(scope="module")
def vllm_client():
    """httpx client pointed at local vLLM. Skip if server unavailable."""
    try:
        resp = httpx.get(f"{VLLM_BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            pytest.skip(f"vLLM server at {VLLM_BASE_URL} returned status {resp.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip(f"vLLM server not reachable at {VLLM_BASE_URL}")

    return httpx.Client(base_url=VLLM_BASE_URL, timeout=120)


def _render_page_png(pdf_path: Path, page_num_1: int, dpi: int = 200) -> bytes:
    """Render a single page as PNG bytes using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num_1 - 1]  # 0-indexed
    pix = page.get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _ocr_with_deepseek(client: httpx.Client, png_bytes: bytes, model: str) -> str:
    """Send a page image to vLLM DeepSeek vision model for OCR."""
    b64_image = base64.b64encode(png_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": "Free OCR.",
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _run_pipeline_page(pdf_path: Path, out_dir: Path, page: int) -> list[TextBlockRecord]:
    """Run the pipeline on a single page and return text blocks.

    Forces CPU mode so Docling doesn't compete with vLLM for GPU memory.
    """
    # Hide CUDA from Docling so it uses CPU (vLLM owns the GPU)
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return _run_pipeline_page_inner(pdf_path, out_dir, page)
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda


def _run_pipeline_page_inner(pdf_path: Path, out_dir: Path, page: int) -> list[TextBlockRecord]:
    """Pipeline execution (runs under CPU-only environment)."""
    config = PipelineConfig(
        pdf_path=pdf_path,
        out_dir=out_dir,
        page_range=str(page),
        force=True,
    )

    document = run_inventory(config)
    doc_id = document.doc_id

    blocks_by_page = extract_with_docling(config, doc_id)
    mupdf_data = extract_with_mupdf(config)

    # Build page heights
    page_heights: dict[int, float] = {}
    for pn, pd in mupdf_data.items():
        page_heights[pn] = pd.height

    blocks_by_page = normalize_blocks(blocks_by_page)
    blocks_by_page, _ = detect_boilerplate(blocks_by_page, page_heights, config)
    blocks_by_page = detect_languages(blocks_by_page, config)
    blocks_by_page = ocr_cleanup_blocks(blocks_by_page)
    blocks_by_page, _ = detect_footnotes(blocks_by_page, page_heights, config, doc_id)

    # Flatten all blocks for the target page
    return blocks_by_page.get(page, [])


def _count_correct(text: str, known_errors: list[tuple[str, str]]) -> tuple[int, list[str]]:
    """Count how many known errors are corrected in the text.

    Returns (count_correct, list_of_correct_words).
    """
    correct = []
    for garbled, fixed in known_errors:
        # Check the correct form is present and the garbled form is NOT
        has_correct = fixed in text
        has_garbled = garbled in text
        if has_correct and not has_garbled:
            correct.append(fixed)
    return len(correct), correct


@pytest.mark.slow
class TestDeepSeekOCRComparison:
    """Compare DeepSeek vision OCR against the pipeline's Docling/MuPDF OCR."""

    def test_page76_ocr_quality(self, swettenham_pdf: Path, vllm_client: httpx.Client, tmp_out_dir: Path):
        """Render Swettenham page 76, OCR with DeepSeek, compare to pipeline text."""
        page_num = 76

        # 1. Render page 76 as PNG
        png_bytes = _render_page_png(swettenham_pdf, page_num)

        # 2. OCR with DeepSeek
        deepseek_text = _ocr_with_deepseek(vllm_client, png_bytes, DEEPSEEK_MODEL)

        # 3. Run pipeline on same page
        pipeline_blocks = _run_pipeline_page(swettenham_pdf, tmp_out_dir, page_num)
        pipeline_text = "\n".join(
            b.text_clean or b.text_raw for b in pipeline_blocks if b.block_type not in ("header", "footer")
        )

        # 4. Print both for manual review
        print("\n" + "=" * 80)
        print("PIPELINE OCR (Docling/MuPDF):")
        print("=" * 80)
        print(pipeline_text[:2000])
        print("\n" + "=" * 80)
        print("DEEPSEEK VISION OCR:")
        print("=" * 80)
        print(deepseek_text[:2000])

        # 5. Compare known errors
        pipeline_correct, pipeline_words = _count_correct(pipeline_text, KNOWN_ERRORS)
        deepseek_correct, deepseek_words = _count_correct(deepseek_text, KNOWN_ERRORS)

        print("\n" + "=" * 80)
        print("KNOWN ERROR COMPARISON:")
        print("=" * 80)
        for garbled, fixed in KNOWN_ERRORS:
            in_pipeline = fixed in pipeline_text and garbled not in pipeline_text
            in_deepseek = fixed in deepseek_text and garbled not in deepseek_text
            status_p = "OK" if in_pipeline else "FAIL"
            status_d = "OK" if in_deepseek else "FAIL"
            print(f"  '{garbled}' → '{fixed}':  pipeline={status_p}  deepseek={status_d}")

        print(f"\nPipeline correct: {pipeline_correct}/{len(KNOWN_ERRORS)} {pipeline_words}")
        print(f"DeepSeek correct: {deepseek_correct}/{len(KNOWN_ERRORS)} {deepseek_words}")

        # 6. Assert DeepSeek gets more right than pipeline
        assert deepseek_correct >= pipeline_correct, (
            f"Expected DeepSeek ({deepseek_correct}) to match or beat pipeline ({pipeline_correct}) "
            f"on known OCR errors"
        )
