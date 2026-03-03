#!/usr/bin/env python3
"""Render a single page from a PDF to PNG using PyMuPDF.

Usage:
    python scripts/render_page.py --pdf PATH --page N --out output.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def render_page(pdf_path: str | Path, page_num: int, out_path: str | Path, dpi: int = 150) -> None:
    """Render a specific page from a PDF to a PNG file.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-based page number.
        out_path: Output PNG file path.
        dpi: Resolution in DPI (default 150).
    """
    doc = fitz.open(str(pdf_path))
    if page_num < 1 or page_num > len(doc):
        raise ValueError(f"Page {page_num} out of range (1-{len(doc)})")

    page = doc[page_num - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(out_path))
    doc.close()
    print(f"Rendered page {page_num} → {out_path} ({pix.width}x{pix.height}px, {dpi} DPI)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PDF page to PNG")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--page", required=True, type=int, help="Page number (1-based)")
    parser.add_argument("--out", default="output.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="DPI (default 150)")
    args = parser.parse_args()

    render_page(args.pdf, args.page, args.out, args.dpi)


if __name__ == "__main__":
    main()
