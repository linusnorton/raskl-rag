"""Pipeline step: extract structured content via Docling."""

from __future__ import annotations

import logging
from pathlib import Path

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox, TextBlockRecord
from ras_docproc.utils.hashing import make_block_id, text_hash

logger = logging.getLogger(__name__)

# Map Docling labels to our block types
DOCLING_LABEL_MAP = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
    "page-header": "header",
    "page-footer": "footer",
    "picture": "paragraph",
    "formula": "paragraph",
    "reference": "paragraph",
}


def _get_block_type(label: str) -> str:
    """Map a Docling label to our block type."""
    return DOCLING_LABEL_MAP.get(label.lower().replace(" ", "_"), "unknown")


def extract_with_docling(config: PipelineConfig, doc_id: str) -> dict[int, list[TextBlockRecord]]:
    """Extract structured text blocks from PDF using Docling.

    Returns a dict mapping page_num_1 -> list of TextBlockRecord.
    """
    from docling.document_converter import DocumentConverter

    pdf_path = config.pdf_path
    logger.info("Running Docling extraction on %s", pdf_path.name)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    page_range = config.parse_page_range()
    max_pages = config.max_pages

    # Get page heights for coordinate conversion (bottom-left → top-left origin)
    page_heights: dict[int, float] = {}
    for page_no, page_obj in doc.pages.items():
        if hasattr(page_obj, "size") and page_obj.size:
            page_heights[page_no] = page_obj.size.height
        else:
            page_heights[page_no] = 792.0  # default letter height

    blocks_by_page: dict[int, list[TextBlockRecord]] = {}
    reading_order = 0

    for item_ix, (item, _level) in enumerate(doc.iterate_items()):
        text = ""
        if hasattr(item, "text"):
            text = item.text or ""

        if not text.strip():
            continue

        # Get provenance info (page, bbox)
        prov_list = getattr(item, "prov", [])
        if not prov_list:
            continue

        prov = prov_list[0]
        page_no = getattr(prov, "page_no", 1)

        # Apply page filtering
        if page_range is not None and (page_no - 1) not in page_range:
            continue
        if max_pages is not None and page_no > max_pages:
            continue

        # Extract bbox and convert from bottom-left to top-left origin
        bbox_obj = getattr(prov, "bbox", None)
        if bbox_obj is not None:
            # Docling bbox uses bottom-left origin: l, t, r, b where t > b
            raw_l = getattr(bbox_obj, "l", 0.0)
            raw_t = getattr(bbox_obj, "t", 0.0)  # top edge (larger y in bottom-left)
            raw_r = getattr(bbox_obj, "r", 0.0)
            raw_b = getattr(bbox_obj, "b", 0.0)  # bottom edge (smaller y in bottom-left)
            page_h = page_heights.get(page_no, 792.0)
            # Convert: new_y = page_height - old_y, and swap so y0 < y1
            bbox = BBox(
                x0=raw_l,
                y0=page_h - raw_t,  # top edge → smaller y in top-left
                x1=raw_r,
                y1=page_h - raw_b,  # bottom edge → larger y in top-left
            )
        else:
            bbox = BBox(x0=0, y0=0, x1=0, y1=0)

        label = getattr(item, "label", "text")
        if hasattr(label, "value"):
            label = label.value
        block_type = _get_block_type(str(label))

        t_hash = text_hash(text)
        bbox_str = f"{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}"
        block_id = make_block_id(doc_id, page_no, bbox_str, block_type, t_hash)

        block = TextBlockRecord(
            block_id=block_id,
            doc_id=doc_id,
            page_num_1=page_no,
            bbox=bbox,
            text_raw=text,
            block_type=block_type,
            reading_order=reading_order,
        )

        blocks_by_page.setdefault(page_no, []).append(block)
        reading_order += 1

    total_blocks = sum(len(v) for v in blocks_by_page.values())
    logger.info("Docling extracted %d blocks across %d pages", total_blocks, len(blocks_by_page))
    return blocks_by_page
