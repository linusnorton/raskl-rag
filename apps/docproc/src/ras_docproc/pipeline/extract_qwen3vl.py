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
JPEG_QUALITY = 85
DPI_FALLBACKS = (200, 150)  # DPI steps to try if JPEG is still too large (unlikely)


def _render_page_to_image_bytes(pdf_path: str, page_index: int, dpi: int) -> tuple[bytes, float, float]:
    """Render a PDF page to JPEG bytes, reducing DPI if needed to stay under Bedrock's size limit.

    Returns (jpeg_bytes, page_width_pts, page_height_pts).
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
            img_bytes = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            logger.debug(
                "Page %d: %d DPI JPEG Q%d → %d bytes (limit %d)",
                page_index + 1,
                current_dpi,
                JPEG_QUALITY,
                len(img_bytes),
                MAX_IMAGE_BYTES,
            )
            if len(img_bytes) <= MAX_IMAGE_BYTES:
                if current_dpi != dpi:
                    logger.info(
                        "Page %d: reduced DPI from %d to %d (%d bytes)",
                        page_index + 1,
                        dpi,
                        current_dpi,
                        len(img_bytes),
                    )
                return img_bytes, page_width, page_height

        # Last resort: use lowest DPI even if still over limit
        logger.warning("Page %d: image still %d bytes at %d DPI", page_index + 1, len(img_bytes), DPI_FALLBACKS[-1])
        return img_bytes, page_width, page_height
    finally:
        doc.close()


_bedrock_clients: dict[str, object] = {}


def _get_bedrock_client(region: str):
    if region not in _bedrock_clients:
        import boto3
        from botocore.config import Config as BotoConfig

        config = BotoConfig(read_timeout=120, retries={"max_attempts": 0})
        _bedrock_clients[region] = boto3.client("bedrock-runtime", region_name=region, config=config)
    return _bedrock_clients[region]


def _call_bedrock(
    region: str,
    model_id: str,
    image_bytes: bytes,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Call Bedrock Converse API with image + formatting prompt. Retries on transient errors."""
    from botocore.exceptions import ClientError, ReadTimeoutError

    client = _get_bedrock_client(region)

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
                                    "format": "jpeg",
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
            elif e.response["Error"]["Code"] in (
                "ThrottlingException",
                "ServiceUnavailableException",
                "InternalServerException",
            ):
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


def _call_model_studio(
    api_key: str,
    base_url: str,
    model_id: str,
    image_bytes: bytes,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Call Alibaba Model Studio via OpenAI-compatible API with image."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {
                                "type": "text",
                                "text": "Convert this page to Markdown following the system instructions.",
                            },
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )

            text = response.choices[0].message.content or ""
            # Strip thinking blocks (Qwen3.5 reasoning)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            return text
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Model Studio: call failed (attempt %d/%d), retrying in %ds: %s",
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
) -> tuple[list[TextBlockRecord], list[str]]:
    """Parse Markdown response into TextBlockRecords.

    No bounding boxes available — use full-page bbox for all blocks.

    Returns:
        Tuple of (blocks, figure_descriptions). figure_descriptions is a list of
        alt-text strings from ![Figure](...) tags found on this page.
    """
    blocks: list[TextBlockRecord] = []
    figure_descriptions: list[str] = []
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

        # Detect ![Figure](description) or ![alt](url) image tags from VL model
        # Description may end with |rotate90cw or |rotate90ccw for rotation
        fig_match = re.match(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$", paragraph)
        if fig_match:
            desc = fig_match.group(2) or fig_match.group(1)
            # Filter out false positives: blank pages, blemishes, etc.
            if desc and not re.search(r"\bblank\b", desc, re.IGNORECASE):
                figure_descriptions.append(desc)
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

            # Convert ^N superscript notation
            text = re.sub(r"\^(\d+)", r"<sup>\1</sup>", line)

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

    return blocks, figure_descriptions


def _process_page(
    config: PipelineConfig,
    doc_id: str,
    page_num: int,
    img_bytes: bytes,
    page_width: float,
    page_height: float,
) -> tuple[int, list[TextBlockRecord], list[str]]:
    """Call the configured VL model and parse response for a single page. Returns (page_num, blocks, figure_descriptions)."""
    if config.llm_provider == "model_studio":
        markdown = _call_model_studio(
            config.model_studio_api_key,
            config.model_studio_base_url,
            config.model_studio_ocr_model_id,
            img_bytes,
            config.qwen3vl_max_tokens,
            config.qwen3vl_system_prompt,
        )
    else:
        markdown = _call_bedrock(
            config.bedrock_region,
            config.bedrock_ocr_model_id,
            img_bytes,
            config.qwen3vl_max_tokens,
            config.qwen3vl_system_prompt,
        )
    logger.debug("VL extraction: page %d response length %d chars", page_num, len(markdown))

    blocks, figure_descriptions = _parse_markdown_to_blocks(markdown, page_width, page_height, doc_id, page_num)
    return page_num, blocks, figure_descriptions


def extract_with_qwen3vl(
    config: PipelineConfig,
    doc_id: str,
) -> tuple[dict[int, list[TextBlockRecord]], dict[int, list[str]]]:
    """Extract structured text blocks from PDF using Bedrock Qwen3 VL.

    Returns a tuple of:
        - dict mapping page_num_1 -> list of TextBlockRecord
        - dict mapping page_num_1 -> list of figure descriptions (from ![Figure](...) tags)

    Page rendering (PyMuPDF) is done sequentially to avoid segfaults with
    certain PDFs under concurrent fitz access. Bedrock API calls are then
    parallelized via ThreadPoolExecutor since they are I/O-bound.
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

    # Render all pages sequentially (PyMuPDF segfaults with some PDFs under concurrent access)
    t0 = time.time()
    rendered: dict[int, tuple[bytes, float, float]] = {}
    for page_num in pages_to_process:
        img_bytes, page_width, page_height = _render_page_to_image_bytes(pdf_path, page_num - 1, config.qwen3vl_dpi)
        rendered[page_num] = (img_bytes, page_width, page_height)
        logger.debug("Qwen3 VL: page %d rendered (%d bytes)", page_num, len(img_bytes))
    render_time = time.time() - t0
    logger.info("Qwen3 VL: rendered %d pages in %.1fs", len(rendered), render_time)

    # Parallelize Bedrock API calls (I/O-bound)
    t0 = time.time()
    blocks_by_page: dict[int, list[TextBlockRecord]] = {}
    figures_by_page: dict[int, list[str]] = {}

    with ThreadPoolExecutor(max_workers=config.qwen3vl_max_workers) as executor:
        futures = {
            executor.submit(_process_page, config, doc_id, page_num, *rendered[page_num]): page_num
            for page_num in pages_to_process
        }

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                pn, blocks, fig_descs = future.result()
                if blocks:
                    blocks_by_page[pn] = blocks
                if fig_descs:
                    figures_by_page[pn] = fig_descs
                logger.info("Qwen3 VL: page %d → %d blocks, %d figures", pn, len(blocks), len(fig_descs))
            except Exception:
                logger.exception("Qwen3 VL: failed on page %d", page_num)
    bedrock_time = time.time() - t0
    logger.info(
        "Qwen3 VL: Bedrock calls for %d pages in %.1fs (max_workers=%d)",
        len(pages_to_process),
        bedrock_time,
        config.qwen3vl_max_workers,
    )

    total_blocks = sum(len(v) for v in blocks_by_page.values())
    if figures_by_page:
        logger.info(
            "Qwen3 VL: detected figures on %d pages: %s",
            len(figures_by_page),
            sorted(figures_by_page.keys()),
        )
    logger.info("Qwen3 VL extracted %d blocks across %d pages", total_blocks, len(blocks_by_page))
    return blocks_by_page, figures_by_page
