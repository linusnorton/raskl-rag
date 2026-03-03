"""Pipeline step: generate HTML debug report with SVG overlays."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import fitz
from jinja2 import Environment, FileSystemLoader, PackageLoader

from ras_docproc.schema import FigureRecord, PageRecord, TextBlockRecord
from ras_docproc.utils.io import ensure_dir, read_jsonl

logger = logging.getLogger(__name__)

BLOCK_TYPE_COLORS = {
    "paragraph": "#3b82f6",   # blue
    "heading": "#22c55e",     # green
    "footnote": "#f97316",    # orange
    "caption": "#a855f7",     # purple
    "header": "#ef4444",      # red
    "footer": "#ef4444",      # red
    "list_item": "#06b6d4",   # cyan
    "table": "#eab308",       # yellow
    "page_number": "#6b7280", # gray
    "unknown": "#9ca3af",     # light gray
}


def generate_report(
    doc_dir: Path,
    output_path: Path,
    page_filter: list[int] | None = None,
    dpi: int = 150,
) -> None:
    """Generate an HTML debug report with page images and SVG overlays.

    Args:
        doc_dir: Path to the document output directory (data/out/{doc_id}/).
        output_path: Path for the output HTML file.
        page_filter: Optional list of page numbers (1-based) to include.
        dpi: DPI for page rendering.
    """
    # Load data
    pages = read_jsonl(doc_dir / "pages.jsonl", PageRecord)
    blocks = read_jsonl(doc_dir / "text_blocks.jsonl", TextBlockRecord)
    figures = read_jsonl(doc_dir / "figures.jsonl", FigureRecord) if (doc_dir / "figures.jsonl").exists() else []

    # Group blocks and figures by page
    blocks_by_page: dict[int, list[TextBlockRecord]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page_num_1, []).append(b)

    figs_by_page: dict[int, list[FigureRecord]] = {}
    for f in figures:
        figs_by_page.setdefault(f.page_num_1, []).append(f)

    # Get source PDF path from documents.jsonl
    from ras_docproc.schema import DocumentRecord
    docs = read_jsonl(doc_dir / "documents.jsonl", DocumentRecord)
    if not docs:
        raise ValueError("No documents.jsonl found")
    pdf_path = docs[0].source_path

    # Render pages
    doc = fitz.open(pdf_path)
    page_data = []

    for pr in pages:
        if page_filter and pr.page_num_1 not in page_filter:
            continue

        page_idx = pr.page_index_0
        if page_idx >= len(doc):
            continue

        page = doc[page_idx]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to base64 PNG for inline embedding
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        scale_x = pix.width / pr.width if pr.width else 1
        scale_y = pix.height / pr.height if pr.height else 1

        # Build SVG overlays for blocks
        svg_rects = []
        page_blocks = blocks_by_page.get(pr.page_num_1, [])
        for block in page_blocks:
            color = BLOCK_TYPE_COLORS.get(block.block_type, "#9ca3af")
            svg_rects.append({
                "x": block.bbox.x0 * scale_x,
                "y": block.bbox.y0 * scale_y,
                "w": block.bbox.width * scale_x,
                "h": block.bbox.height * scale_y,
                "color": color,
                "type": block.block_type,
                "text_preview": (block.text_clean or block.text_raw)[:80],
                "block_id": block.block_id[:12],
                "lang": block.lang or "",
            })

        # Figure overlays
        fig_rects = []
        for fig in figs_by_page.get(pr.page_num_1, []):
            if fig.bbox:
                fig_rects.append({
                    "x": fig.bbox.x0 * scale_x,
                    "y": fig.bbox.y0 * scale_y,
                    "w": fig.bbox.width * scale_x,
                    "h": fig.bbox.height * scale_y,
                    "figure_id": fig.figure_id[:12],
                    "caption": fig.caption_text_clean[:60] if fig.caption_text_clean else "",
                })

        # Footnote zone line
        fn_zone_y = pr.height * 0.72 * scale_y

        page_data.append({
            "page_num": pr.page_num_1,
            "width": pix.width,
            "height": pix.height,
            "img_b64": img_b64,
            "svg_rects": svg_rects,
            "fig_rects": fig_rects,
            "fn_zone_y": fn_zone_y,
            "blocks": page_blocks,
            "has_vertical_text": pr.has_vertical_text,
            "suggested_rotation": pr.suggested_rotation_cw,
            "vertical_text_ratio": pr.vertical_text_ratio,
        })

    doc.close()

    # Render template
    try:
        env = Environment(loader=PackageLoader("ras_docproc", "templates"))
    except Exception:
        template_dir = Path(__file__).resolve().parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))

    template = env.get_template("report.html.j2")
    html = template.render(
        doc_id=docs[0].doc_id,
        source_filename=docs[0].source_filename,
        pages=page_data,
        block_type_colors=BLOCK_TYPE_COLORS,
    )

    ensure_dir(output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s (%d pages)", output_path, len(page_data))
