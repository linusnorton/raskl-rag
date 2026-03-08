"""Pipeline step: extract and process embedded figures/images."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from PIL import Image

from ras_docproc.config import PipelineConfig
from ras_docproc.pipeline.extract_mupdf import MuPDFPageData
from ras_docproc.schema import FigureRecord
from ras_docproc.utils.hashing import make_block_id
from ras_docproc.utils.io import ensure_dir

logger = logging.getLogger(__name__)

THUMB_SIZE = (200, 200)
FULL_PAGE_AREA_THRESHOLD = 0.90  # Skip images covering ≥90% of page area


def detect_figures(
    mupdf_data: dict[int, MuPDFPageData],
    config: PipelineConfig,
    doc_id: str,
    page_rotations: dict[int, int] | None = None,
) -> list[FigureRecord]:
    """Extract embedded images and produce normalized JPG + thumbnails.

    For pages with suggested rotation, renders the page and rotates the clip.

    Args:
        mupdf_data: Per-page MuPDF extraction data.
        config: Pipeline configuration.
        doc_id: Document ID.
        page_rotations: Optional dict of page_num -> suggested_rotation_cw.

    Returns:
        List of FigureRecord.
    """
    assets_dir = ensure_dir(config.out_dir / "out" / doc_id / "assets")
    page_rotations = page_rotations or {}
    figures: list[FigureRecord] = []

    skipped_scans = 0
    for page_num, page_data in mupdf_data.items():
        page_area = page_data.width * page_data.height
        for img_idx, img_info in enumerate(page_data.images):
            # Filter out full-page scan images (background scans in scanned PDFs)
            if page_area > 0 and img_info.bbox:
                img_area = img_info.bbox.area
                if img_area >= FULL_PAGE_AREA_THRESHOLD * page_area:
                    skipped_scans += 1
                    continue

            # Save original image bytes
            ext = img_info.ext or "png"
            sha256 = hashlib.sha256(img_info.image_bytes).hexdigest()

            orig_path = assets_dir / f"p{page_num}_img{img_idx}_orig.{ext}"
            orig_path.write_bytes(img_info.image_bytes)

            # Convert to normalized JPG
            try:
                pil_img = Image.open(io.BytesIO(img_info.image_bytes))
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB")

                jpg_path = assets_dir / f"p{page_num}_img{img_idx}.jpg"
                pil_img.save(str(jpg_path), "JPEG", quality=85)

                # Thumbnail
                thumb = pil_img.copy()
                thumb.thumbnail(THUMB_SIZE)
                thumb_path = assets_dir / f"p{page_num}_img{img_idx}_thumb.jpg"
                thumb.save(str(thumb_path), "JPEG", quality=80)
            except Exception:
                logger.warning("Failed to process image p%d img%d", page_num, img_idx, exc_info=True)
                jpg_path = orig_path
                thumb_path = orig_path

            fig_id = make_block_id(doc_id, page_num, f"fig{img_idx}", "figure", sha256[:16])

            figures.append(
                FigureRecord(
                    figure_id=fig_id,
                    doc_id=doc_id,
                    page_num_1=page_num,
                    figure_index_on_page=img_idx,
                    bbox=img_info.bbox,
                    asset_original_path=str(orig_path),
                    asset_jpg_path=str(jpg_path),
                    asset_thumb_path=str(thumb_path),
                    asset_sha256=sha256,
                )
            )

        # Handle pages with suggested rotation
        rotation = page_rotations.get(page_num, 0)
        if rotation != 0 and page_data.vertical_line_count > 0:
            rendered_fig = _render_rotated_page(config.pdf_path, page_num, rotation, assets_dir, doc_id)
            if rendered_fig:
                figures.append(rendered_fig)

    if skipped_scans > 0:
        logger.info("Skipped %d full-page scan images", skipped_scans)
    logger.info("Extracted %d figures across %d pages", len(figures), len(mupdf_data))
    return figures


def _render_rotated_page(
    pdf_path: Path,
    page_num: int,
    rotation_cw: int,
    assets_dir: Path,
    doc_id: str,
) -> FigureRecord | None:
    """Render a page at high DPI, rotate, and save as a figure."""
    import fitz

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        doc.close()

        pil_img = Image.open(io.BytesIO(img_bytes))
        # PIL rotate is counter-clockwise, so negate for CW
        rotated = pil_img.rotate(-rotation_cw, expand=True)

        jpg_path = assets_dir / f"p{page_num}_rendered_rot{rotation_cw}.jpg"
        rotated_rgb = rotated.convert("RGB") if rotated.mode != "RGB" else rotated
        rotated_rgb.save(str(jpg_path), "JPEG", quality=90)

        thumb = rotated_rgb.copy()
        thumb.thumbnail(THUMB_SIZE)
        thumb_path = assets_dir / f"p{page_num}_rendered_rot{rotation_cw}_thumb.jpg"
        thumb.save(str(thumb_path), "JPEG", quality=80)

        sha256 = hashlib.sha256(jpg_path.read_bytes()).hexdigest()
        fig_id = make_block_id(doc_id, page_num, f"rendered_rot{rotation_cw}", "figure", sha256[:16])

        return FigureRecord(
            figure_id=fig_id,
            doc_id=doc_id,
            page_num_1=page_num,
            figure_index_on_page=99,
            asset_jpg_path=str(jpg_path),
            asset_thumb_path=str(thumb_path),
            asset_sha256=sha256,
            derived_from="rendered_clip",
            applied_rotation_cw=rotation_cw,
        )
    except Exception:
        logger.warning("Failed to render rotated page %d", page_num, exc_info=True)
        return None
