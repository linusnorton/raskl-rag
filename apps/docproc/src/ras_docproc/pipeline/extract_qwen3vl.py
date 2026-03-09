"""Pipeline step: extract structured content via Bedrock Qwen3 VL vision model."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds


MAX_IMAGE_BYTES = 3_700_000  # 3.7 MB — just under Bedrock's 3.75 MB per-image limit
DPI_FALLBACKS = (200, 150, 120)  # DPI steps to try if the image is too large


def _render_page_to_png_bytes(pdf_path: str, page_index: int, dpi: int) -> tuple[bytes, float, float]:
    """Render a PDF page to PNG bytes, reducing DPI if needed to stay under Bedrock's size limit.

    Returns (png_bytes, page_width_pts, page_height_pts).
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height

        for current_dpi in (dpi, *DPI_FALLBACKS):
            zoom = current_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            logger.debug("Page %d: %d DPI → %d bytes (limit %d)", page_index + 1, current_dpi, len(png_bytes), MAX_IMAGE_BYTES)
            if len(png_bytes) <= MAX_IMAGE_BYTES:
                if current_dpi != dpi:
                    logger.info("Page %d: reduced DPI from %d to %d (%d bytes)", page_index + 1, dpi, current_dpi, len(png_bytes))
                return png_bytes, page_width, page_height

        # Last resort: use lowest DPI even if still over limit
        logger.warning("Page %d: image still %d bytes at %d DPI", page_index + 1, len(png_bytes), DPI_FALLBACKS[-1])
        return png_bytes, page_width, page_height
    finally:
        doc.close()


def _call_bedrock(
    region: str,
    model_id: str,
    image_bytes: bytes,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Call Bedrock Converse API with image + formatting prompt. Retries on transient errors."""
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, ReadTimeoutError

    bedrock_config = BotoConfig(read_timeout=120, retries={"max_attempts": 0})
    client = boto3.client("bedrock-runtime", region_name=region, config=bedrock_config)

    for attempt in range(MAX_RETRIES):
        try:
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
                system=[{"text": system_prompt}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )

            text = response["output"]["message"]["content"][0]["text"]
            # Strip markdown code fences that Qwen3 VL sometimes wraps around its response
            text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            return text
        except (ReadTimeoutError, ClientError) as e:
            # Don't retry non-transient errors (auth, access denied, payload too large)
            if isinstance(e, ReadTimeoutError):
                retryable = True
            elif e.response["Error"]["Code"] in ("ThrottlingException", "ServiceUnavailableException", "InternalServerException"):
                retryable = True
            elif e.response["Error"]["Code"] == "ValidationException" and "limit exceeded" not in str(e):
                retryable = True
            else:
                retryable = False
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Qwen3 VL: Bedrock call failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                e,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


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
        config.qwen3vl_system_prompt,
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
        futures = {executor.submit(_process_page, config, doc_id, page_num): page_num for page_num in pages_to_process}

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
