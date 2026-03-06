"""Pipeline step: extract structured content via Bedrock Qwen3 VL vision model."""

from __future__ import annotations

import base64
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a document OCR engine. Convert the page image to clean Markdown text.

Rules:
- Preserve the reading order exactly as it appears on the page.
- Use **bold** and *italic* for emphasis where the original uses bold/italic.
- Use # for main headings, ## for subheadings.
- Use > for block quotes.
- If the page has footnotes (small text at the bottom, often after a horizontal rule), \
separate them with --- and format each as a numbered line: 1. footnote text
- Convert superscript footnote references in body text to ^N notation (e.g. ^1, ^23).
- Do NOT describe images or figures — skip them entirely.
- Do NOT add any commentary, explanation, or preamble. Output ONLY the Markdown text."""


def _render_page_to_png_bytes(pdf_path: str, page_index: int, dpi: int) -> tuple[bytes, float, float]:
    """Render a PDF page to PNG bytes. Returns (png_bytes, page_width_pts, page_height_pts)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png"), page_width, page_height
    finally:
        doc.close()


def _call_bedrock(
    region: str,
    model_id: str,
    image_bytes: bytes,
    max_tokens: int,
) -> str:
    """Call Bedrock Converse API with image + formatting prompt."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_bytes},
                        },
                    },
                    {"text": "Convert this page to Markdown following the system instructions."},
                ],
            },
        ],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )

    text = response["output"]["message"]["content"][0]["text"]
    # Strip markdown code fences that Qwen3 VL sometimes wraps around its response
    text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def _parse_markdown_to_blocks(
    markdown: str,
    page_width: float,
    page_height: float,
    doc_id: str,
    page_num: int,
) -> list[TextBlockRecord]:
    """Parse Markdown response into TextBlockRecords.

    No bounding boxes available — use full-page bbox for all blocks.
    """
    blocks: list[TextBlockRecord] = []
    full_bbox = BBox(x0=0, y0=0, x1=page_width, y1=page_height)

    # Split into sections: main content and footnotes (separated by ---)
    parts = re.split(r"\n---+\n", markdown, maxsplit=1)
    main_text = parts[0]
    footnote_text = parts[1] if len(parts) > 1 else ""

    reading_order = 0

    # Parse main content paragraphs
    for paragraph in re.split(r"\n{2,}", main_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Determine block type
        if paragraph.startswith("#"):
            block_type = "heading"
            # Strip heading markers for text_raw but keep formatting
            text = re.sub(r"^#+\s*", "", paragraph)
        elif paragraph.startswith(">"):
            block_type = "paragraph"
            text = re.sub(r"^>\s*", "", paragraph, flags=re.MULTILINE)
        elif re.match(r"^[-*]\s+", paragraph) or re.match(r"^\d+\.\s+", paragraph):
            block_type = "list_item"
            text = paragraph
        else:
            block_type = "paragraph"
            text = paragraph

        # Convert ^N superscript notation to <sup>N</sup>
        text = re.sub(r"\^(\d+)", r"<sup>\1</sup>", text)

        if not text.strip():
            continue

        t_hash = text_hash(text)
        bbox_str = f"0.0,0.0,{page_width:.1f},{page_height:.1f}"
        block_id = make_block_id(doc_id, page_num, bbox_str, block_type, t_hash)

        blocks.append(
            TextBlockRecord(
                block_id=block_id,
                doc_id=doc_id,
                page_num_1=page_num,
                bbox=full_bbox,
                text_raw=text,
                block_type=block_type,
                reading_order=reading_order,
            )
        )
        reading_order += 1

    # Parse footnotes
    if footnote_text.strip():
        for line in footnote_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            # Match numbered footnote: "1. text" or "1 text"
            m = re.match(r"^(\d+)[.\s]+(.+)", line)
            if m:
                text = m.group(2).strip()
                # Convert ^N superscript notation
                text = re.sub(r"\^(\d+)", r"<sup>\1</sup>", text)
            else:
                text = line

            if not text:
                continue

            t_hash = text_hash(text)
            bbox_str = f"0.0,0.0,{page_width:.1f},{page_height:.1f}"
            block_id = make_block_id(doc_id, page_num, bbox_str, "footnote", t_hash)

            blocks.append(
                TextBlockRecord(
                    block_id=block_id,
                    doc_id=doc_id,
                    page_num_1=page_num,
                    bbox=full_bbox,
                    text_raw=text,
                    block_type="footnote",
                    reading_order=reading_order,
                )
            )
            reading_order += 1

    return blocks


def _process_page(
    config: PipelineConfig,
    doc_id: str,
    page_num: int,
) -> tuple[int, list[TextBlockRecord]]:
    """Process a single page: render → Bedrock call → parse. Returns (page_num, blocks)."""
    page_idx = page_num - 1
    pdf_path = str(config.pdf_path)

    png_bytes, page_width, page_height = _render_page_to_png_bytes(pdf_path, page_idx, config.qwen3vl_dpi)
    logger.debug("Qwen3 VL: page %d rendered (%d bytes)", page_num, len(png_bytes))

    markdown = _call_bedrock(
        config.bedrock_region,
        config.bedrock_ocr_model_id,
        png_bytes,
        config.qwen3vl_max_tokens,
    )
    logger.debug("Qwen3 VL: page %d response length %d chars", page_num, len(markdown))

    blocks = _parse_markdown_to_blocks(markdown, page_width, page_height, doc_id, page_num)
    return page_num, blocks


def extract_with_qwen3vl(config: PipelineConfig, doc_id: str) -> dict[int, list[TextBlockRecord]]:
    """Extract structured text blocks from PDF using Bedrock Qwen3 VL.

    Returns a dict mapping page_num_1 -> list of TextBlockRecord.
    Pages are processed in parallel using ThreadPoolExecutor.
    """
    logger.info("Running Qwen3 VL extraction on %s (model=%s)", config.pdf_path.name, config.bedrock_ocr_model_id)

    pdf_path = str(config.pdf_path)
    page_range = config.parse_page_range()
    max_pages = config.max_pages

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    pages_to_process: list[int] = []
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if page_range is not None and page_idx not in page_range:
            continue
        if max_pages is not None and page_num > max_pages:
            continue
        pages_to_process.append(page_num)

    blocks_by_page: dict[int, list[TextBlockRecord]] = {}

    with ThreadPoolExecutor(max_workers=config.qwen3vl_max_workers) as executor:
        futures = {
            executor.submit(_process_page, config, doc_id, page_num): page_num
            for page_num in pages_to_process
        }

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                pn, blocks = future.result()
                if blocks:
                    blocks_by_page[pn] = blocks
                logger.info("Qwen3 VL: page %d → %d blocks", pn, len(blocks))
            except Exception:
                logger.exception("Qwen3 VL: failed on page %d", page_num)

    total_blocks = sum(len(v) for v in blocks_by_page.values())
    logger.info("Qwen3 VL extracted %d blocks across %d pages", total_blocks, len(blocks_by_page))
    return blocks_by_page
