"""Pipeline step: low-level extraction via PyMuPDF (fitz)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import BBox

logger = logging.getLogger(__name__)


@dataclass
class SpanInfo:
    """Info about a text span for superscript/direction detection."""

    text: str
    font_size: float
    bbox: BBox
    font_name: str = ""
    flags: int = 0
    direction: tuple[float, float] = (1.0, 0.0)  # (cos, sin) of text direction


@dataclass
class ImageInfo:
    """Info about an embedded image on a page."""

    xref: int
    bbox: BBox
    width: int
    height: int
    image_bytes: bytes = field(repr=False)
    ext: str = "png"


@dataclass
class MuPDFPageData:
    """All data extracted from a page via MuPDF."""

    page_num_1: int
    width: float
    height: float
    rotation: int
    spans: list[SpanInfo] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    vertical_line_count: int = 0
    total_line_count: int = 0


def extract_pdf_metadata(config: PipelineConfig) -> dict[str, str]:
    """Extract PDF-level metadata via PyMuPDF.

    Returns the doc.metadata dict (title, author, subject, keywords, creator, producer, etc.).
    """
    doc = fitz.open(str(config.pdf_path))
    metadata = dict(doc.metadata) if doc.metadata else {}
    metadata["page_count"] = str(doc.page_count)
    doc.close()
    return metadata


def extract_with_mupdf(config: PipelineConfig) -> dict[int, MuPDFPageData]:
    """Extract span-level text data and images from PDF via PyMuPDF.

    Returns dict mapping page_num_1 -> MuPDFPageData.
    """
    pdf_path = config.pdf_path
    logger.info("Running MuPDF extraction on %s", pdf_path.name)

    doc = fitz.open(str(pdf_path))
    page_range = config.parse_page_range()
    max_pages = config.max_pages

    result: dict[int, MuPDFPageData] = {}

    for page_idx in range(len(doc)):
        page_num = page_idx + 1

        if page_range is not None and page_idx not in page_range:
            continue
        if max_pages is not None and page_num > max_pages:
            break

        page = doc[page_idx]
        rect = page.rect

        page_data = MuPDFPageData(
            page_num_1=page_num,
            width=rect.width,
            height=rect.height,
            rotation=page.rotation,
        )

        # Extract text with dict mode for span-level info
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                # Track line direction
                line_dir = line.get("dir", (1.0, 0.0))
                page_data.total_line_count += 1
                # Vertical text: dir ≈ (0, ±1)
                if abs(line_dir[0]) < 0.3 and abs(line_dir[1]) > 0.7:
                    page_data.vertical_line_count += 1

                for span in line.get("spans", []):
                    sbbox = span.get("bbox", (0, 0, 0, 0))
                    page_data.spans.append(
                        SpanInfo(
                            text=span.get("text", ""),
                            font_size=span.get("size", 0),
                            bbox=BBox(x0=sbbox[0], y0=sbbox[1], x1=sbbox[2], y1=sbbox[3]),
                            font_name=span.get("font", ""),
                            flags=span.get("flags", 0),
                            direction=line_dir,
                        )
                    )

        # Extract embedded images
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Get image bbox on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    r = img_rects[0]
                    bbox = BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                else:
                    bbox = BBox(x0=0, y0=0, x1=rect.width, y1=rect.height)

                page_data.images.append(
                    ImageInfo(
                        xref=xref,
                        bbox=bbox,
                        width=width,
                        height=height,
                        image_bytes=image_bytes,
                        ext=ext,
                    )
                )
            except Exception:
                logger.warning("Failed to extract image xref=%d on page %d", xref, page_num, exc_info=True)

        result[page_num] = page_data

    doc.close()
    total_spans = sum(len(pd.spans) for pd in result.values())
    total_images = sum(len(pd.images) for pd in result.values())
    logger.info("MuPDF extracted %d spans, %d images across %d pages", total_spans, total_images, len(result))
    return result
