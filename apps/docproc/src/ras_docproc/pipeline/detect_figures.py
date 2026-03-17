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
FULL_PAGE_AREA_THRESHOLD = 0.90  # Skip images covering ≥90% of page area (pages with text spans)
SCAN_BG_THRESHOLD = 0.70  # On pages with no text spans, images ≥70% are scan backgrounds


def detect_figures(
    mupdf_data: dict[int, MuPDFPageData],
    config: PipelineConfig,
    doc_id: str,
    page_rotations: dict[int, int] | None = None,
    vl_figure_pages: dict[int, list[str]] | None = None,
) -> list[FigureRecord]:
    """Extract embedded images and produce normalized JPG + thumbnails.

    For pages with suggested rotation, renders the page and rotates the clip.
    For scanned PDFs where the VL model detected illustrations on a page,
    renders the full page as a figure instead of skipping it.

    Args:
        mupdf_data: Per-page MuPDF extraction data.
        config: Pipeline configuration.
        doc_id: Document ID.
        page_rotations: Optional dict of page_num -> suggested_rotation_cw.
        vl_figure_pages: Optional dict of page_num -> list of figure descriptions
            detected by the VL model (from ![Figure](...) tags).

    Returns:
        List of FigureRecord.
    """
    assets_dir = ensure_dir(config.out_dir / "out" / doc_id / "assets")
    page_rotations = page_rotations or {}
    vl_figure_pages = vl_figure_pages or {}
    figures: list[FigureRecord] = []

    skipped_scans = 0
    rendered_scans = 0
    rendered_scan_pages: set[int] = set()  # Track pages already rendered to avoid duplicates

    # Pages with no MuPDF text spans are pure image scans (no OCR layer).
    # Use a lower threshold to skip scan background images on these pages.
    textless_pages = {pn for pn, pd in mupdf_data.items() if not pd.spans}

    # Pre-check which pages have non-fullpage images (i.e. real embedded figures).
    # If a page has real figures, don't render the full-page scan — the real figures are better.
    pages_with_real_images: set[int] = set()
    for page_num, page_data in mupdf_data.items():
        page_area = page_data.width * page_data.height
        threshold = SCAN_BG_THRESHOLD if page_num in textless_pages else FULL_PAGE_AREA_THRESHOLD
        for img_info in page_data.images:
            if page_area > 0 and img_info.bbox:
                if img_info.bbox.area < threshold * page_area:
                    pages_with_real_images.add(page_num)
                    break

    for page_num, page_data in mupdf_data.items():
        page_area = page_data.width * page_data.height
        for img_idx, img_info in enumerate(page_data.images):
            # Filter out full-page scan images (background scans in scanned PDFs)
            if page_area > 0 and img_info.bbox:
                img_area = img_info.bbox.area
                threshold = SCAN_BG_THRESHOLD if page_num in textless_pages else FULL_PAGE_AREA_THRESHOLD
                if img_area >= threshold * page_area:
                    # If the VL model detected a figure on this page AND there are no
                    # real embedded images, render the scan as a figure.
                    # Skip if the page already has real images — they'll be extracted normally.
                    # Only trust VL figure tags on textless pages — on pages with OCR text,
                    # VL often false-positives on text that describes figures on other pages.
                    if (
                        page_num in vl_figure_pages
                        and page_num not in rendered_scan_pages
                        and page_num not in pages_with_real_images
                        and page_num in textless_pages
                    ):
                        rendered_scan_pages.add(page_num)
                        rotation = page_rotations.get(page_num, 0)
                        rendered_fig = _render_scan_page_figure(
                            config.pdf_path, page_num, assets_dir, doc_id, vl_figure_pages[page_num], rotation,
                        )
                        if rendered_fig:
                            figures.append(rendered_fig)
                            rendered_scans += 1
                    else:
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

    if textless_pages:
        logger.info("Detected %d textless pages (pure image scans, threshold=%.0f%%)", len(textless_pages), SCAN_BG_THRESHOLD * 100)
    if skipped_scans > 0:
        logger.info("Skipped %d full-page scan images", skipped_scans)
    if rendered_scans > 0:
        logger.info("Rendered %d full-page scans as figures (VL-detected illustrations)", rendered_scans)
    logger.info("Extracted %d figures across %d pages", len(figures), len(mupdf_data))
    return figures


def _parse_vl_rotation(descriptions: list[str]) -> tuple[list[str], int]:
    """Parse rotation hints from VL figure descriptions.

    Descriptions may end with |rotate90cw or |rotate90ccw.
    Returns (clean_descriptions, rotation_cw).
    """
    import re

    clean = []
    rotation = 0
    for desc in descriptions:
        m = re.search(r"\|(rotate90cw|rotate90ccw)\s*$", desc, re.IGNORECASE)
        if m:
            if m.group(1).lower() == "rotate90cw":
                rotation = 90
            else:
                rotation = 270
            desc = desc[: m.start()].rstrip()
        clean.append(desc)
    return clean, rotation


def _render_scan_page_figure(
    pdf_path: Path,
    page_num: int,
    assets_dir: Path,
    doc_id: str,
    descriptions: list[str],
    rotation_cw: int = 0,
) -> FigureRecord | None:
    """Render a full-page scan as a figure when the VL model detected an illustration.

    Args:
        rotation_cw: Clockwise rotation from the rotation detection stage. Applied to
            correct sideways illustrations in scanned PDFs.
    """
    import fitz

    try:
        # Parse rotation hints from VL descriptions (e.g. "desc|rotate90cw")
        clean_descs, vl_rotation = _parse_vl_rotation(descriptions)

        # Use rotation detection stage rotation first; fall back to VL hint
        effective_rotation = rotation_cw if rotation_cw != 0 else vl_rotation

        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        doc.close()

        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")

        # Apply rotation to correct sideways illustrations
        if effective_rotation != 0:
            pil_img = pil_img.rotate(-effective_rotation, expand=True)
            logger.info("Rotated scan page %d by %d° CW", page_num, effective_rotation)

        jpg_path = assets_dir / f"p{page_num}_scan_figure.jpg"
        pil_img.save(str(jpg_path), "JPEG", quality=90)

        thumb = pil_img.copy()
        thumb.thumbnail(THUMB_SIZE)
        thumb_path = assets_dir / f"p{page_num}_scan_figure_thumb.jpg"
        thumb.save(str(thumb_path), "JPEG", quality=80)

        sha256 = hashlib.sha256(jpg_path.read_bytes()).hexdigest()
        fig_id = make_block_id(doc_id, page_num, "scan_figure", "figure", sha256[:16])

        caption = "; ".join(clean_descs)

        return FigureRecord(
            figure_id=fig_id,
            doc_id=doc_id,
            page_num_1=page_num,
            figure_index_on_page=0,
            asset_jpg_path=str(jpg_path),
            asset_thumb_path=str(thumb_path),
            asset_sha256=sha256,
            caption_text_raw=caption,
            derived_from="vl_detected_scan",
            applied_rotation_cw=effective_rotation,
        )
    except Exception:
        logger.warning("Failed to render scan page %d as figure", page_num, exc_info=True)
        return None


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
