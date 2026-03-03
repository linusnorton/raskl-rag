"""Pipeline step: extract structured content via DeepSeek-OCR grounding mode."""

from __future__ import annotations

import base64
import logging
import re

import fitz
import httpx

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Regex to parse tag lines: tag_type[[x1, y1, x2, y2]]
TAG_LINE_RE = re.compile(
    r"^(\w+)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]",
    re.MULTILINE,
)

# Map DeepSeek tag types to our block types
TAG_TYPE_MAP: dict[str, str | None] = {
    "title": "heading",
    "sub_title": "heading",
    "text": None,  # further classify via _classify_block_type
    "image_caption": "paragraph",
    "image": None,  # skip — no textual content
}


def _classify_block_type(text: str) -> str:
    """Classify block type from markdown markers in the text."""
    stripped = text.strip()
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
        return "list_item"
    if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
        return "table"
    return "paragraph"


def _strip_markdown_markers(text: str) -> str:
    """Remove leading markdown markers from text for clean output."""
    stripped = text.strip()
    # Strip heading markers
    if stripped.startswith("#"):
        stripped = re.sub(r"^#+\s*", "", stripped)
    # Strip list markers
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    stripped = re.sub(r"^\d+\.\s+", "", stripped)
    return stripped.strip()


def _parse_grounded_markdown(
    response_text: str,
    page_width: float,
    page_height: float,
    doc_id: str,
    page_num: int,
) -> list[TextBlockRecord]:
    """Parse DeepSeek-OCR grounding output into TextBlockRecords.

    Grounding format: tag_type[[x1, y1, x2, y2]]\\ncontent...
    Coordinates are normalized to 0-999 range.
    """
    blocks: list[TextBlockRecord] = []
    reading_order = 0

    # Collect all tag positions
    matches = list(TAG_LINE_RE.finditer(response_text))
    if not matches:
        return blocks

    for i, match in enumerate(matches):
        tag_type = match.group(1)
        nx1, ny1, nx2, ny2 = int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))

        # Skip image tags — no textual content
        if tag_type == "image":
            continue

        # Extract content between end of this tag line and start of next tag (or end of string)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
        raw_text = response_text[content_start:content_end].strip()

        # Skip empty text
        if not raw_text:
            continue

        # Determine block type from tag, falling back to markdown classification for "text" tags
        mapped = TAG_TYPE_MAP.get(tag_type)
        if mapped is not None:
            block_type = mapped
        else:
            block_type = _classify_block_type(raw_text)

        # Strip markdown markers for the raw text
        clean = _strip_markdown_markers(raw_text)
        if not clean:
            continue

        # Denormalize coordinates from 0-999 to page dimensions
        x0 = (nx1 / 999.0) * page_width
        y0 = (ny1 / 999.0) * page_height
        x1 = (nx2 / 999.0) * page_width
        y1 = (ny2 / 999.0) * page_height

        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)

        t_hash = text_hash(clean)
        bbox_str = f"{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}"
        block_id = make_block_id(doc_id, page_num, bbox_str, block_type, t_hash)

        blocks.append(
            TextBlockRecord(
                block_id=block_id,
                doc_id=doc_id,
                page_num_1=page_num,
                bbox=bbox,
                text_raw=clean,
                block_type=block_type,
                reading_order=reading_order,
            )
        )
        reading_order += 1

    return blocks


def _render_page_to_base64(pdf_path: str, page_index: int, dpi: int) -> tuple[str, float, float]:
    """Render a PDF page to a base64-encoded PNG string.

    Returns (base64_string, page_width_pts, page_height_pts).
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return b64, page_width, page_height
    finally:
        doc.close()


def _call_vllm(
    base_url: str,
    model: str,
    image_b64: str,
    max_tokens: int,
) -> str:
    """Call vLLM chat completions endpoint with a grounding prompt."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    response = httpx.post(url, json=payload, timeout=120.0)
    if response.status_code != 200:
        logger.error("vLLM error %d: %s", response.status_code, response.text)
        response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _check_vllm_health(base_url: str) -> None:
    """Verify vLLM server is reachable. Raises RuntimeError if not."""
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        raise RuntimeError(
            f"vLLM server not reachable at {base_url}. "
            "Start the server with: vllm serve deepseek-ai/DeepSeek-OCR"
        ) from exc


def extract_with_deepseek(config: PipelineConfig, doc_id: str) -> dict[int, list[TextBlockRecord]]:
    """Extract structured text blocks from PDF using DeepSeek-OCR via vLLM.

    Returns a dict mapping page_num_1 -> list of TextBlockRecord.
    """
    logger.info("Running DeepSeek-OCR extraction on %s", config.pdf_path.name)

    # Health check
    _check_vllm_health(config.vllm_base_url)

    pdf_path = str(config.pdf_path)
    page_range = config.parse_page_range()
    max_pages = config.max_pages

    # Determine which pages to process
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    pages_to_process: list[int] = []
    for page_idx in range(total_pages):
        page_num = page_idx + 1  # 1-based
        if page_range is not None and page_idx not in page_range:
            continue
        if max_pages is not None and page_num > max_pages:
            continue
        pages_to_process.append(page_num)

    blocks_by_page: dict[int, list[TextBlockRecord]] = {}

    for page_num in pages_to_process:
        page_idx = page_num - 1
        logger.debug("DeepSeek-OCR: processing page %d", page_num)

        # Render page to PNG
        image_b64, page_width, page_height = _render_page_to_base64(pdf_path, page_idx, config.deepseek_dpi)

        # Call vLLM
        response_text = _call_vllm(config.vllm_base_url, config.vllm_model, image_b64, config.deepseek_max_tokens)

        # Parse grounding output
        blocks = _parse_grounded_markdown(response_text, page_width, page_height, doc_id, page_num)

        if blocks:
            blocks_by_page[page_num] = blocks
        else:
            # Fallback: no grounding tags found, use full response as single paragraph
            text = response_text.strip()
            if text:
                logger.warning("Page %d: no grounding tags, falling back to single paragraph block", page_num)
                bbox = BBox(x0=0, y0=0, x1=page_width, y1=page_height)
                t_hash = text_hash(text)
                bbox_str = f"0.0,0.0,{page_width:.1f},{page_height:.1f}"
                block_id = make_block_id(doc_id, page_num, bbox_str, "paragraph", t_hash)
                blocks_by_page[page_num] = [
                    TextBlockRecord(
                        block_id=block_id,
                        doc_id=doc_id,
                        page_num_1=page_num,
                        bbox=bbox,
                        text_raw=text,
                        block_type="paragraph",
                        reading_order=0,
                    )
                ]

    total_blocks = sum(len(v) for v in blocks_by_page.values())
    logger.info("DeepSeek-OCR extracted %d blocks across %d pages", total_blocks, len(blocks_by_page))
    return blocks_by_page
