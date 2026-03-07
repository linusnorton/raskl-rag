"""CLI for the document processing pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """ras-docproc: process PDFs into structured JSONL for RAG."""
    _setup_logging(verbose)


def _run_pipeline(pdf_path: Path, out_dir: Path, max_pages: int | None, page_range: str | None, force: bool, backend: str | None) -> str:
    """Run the full pipeline on a single PDF. Returns the doc_id."""
    from tqdm import tqdm

    from ras_docproc.config import PipelineConfig
    from ras_docproc.pipeline.boilerplate import detect_boilerplate
    from ras_docproc.pipeline.detect_captions import detect_captions
    from ras_docproc.pipeline.detect_content_area import detect_content_area
    from ras_docproc.pipeline.detect_figures import detect_figures
    from ras_docproc.pipeline.detect_footnotes import detect_footnotes
    from ras_docproc.pipeline.detect_language import detect_languages
    from ras_docproc.pipeline.detect_rotation import detect_rotation
    from ras_docproc.pipeline.export_jsonl import export_all
    from ras_docproc.pipeline.extract_deepseek import extract_with_deepseek
    from ras_docproc.pipeline.extract_docling import extract_with_docling
    from ras_docproc.pipeline.extract_qwen3vl import extract_with_qwen3vl
    from ras_docproc.pipeline.extract_metadata import extract_metadata
    from ras_docproc.pipeline.extract_mupdf import extract_pdf_metadata, extract_with_mupdf
    from ras_docproc.pipeline.inventory import run_inventory
    from ras_docproc.pipeline.link_footnote_refs import apply_ref_markup, link_footnote_refs
    from ras_docproc.pipeline.normalize_text import normalize_blocks, ocr_cleanup_blocks
    from ras_docproc.schema import PageRecord
    from ras_docproc.utils.hashing import page_content_hash, text_hash

    # Auto-detect backend: messy/ → qwen3vl, otherwise docling
    if backend is None:
        backend = "qwen3vl" if "/messy/" in str(pdf_path) else "docling"

    config = PipelineConfig(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=max_pages,
        page_range=page_range,
        force=force,
        extraction_backend=backend,
    )

    extract_labels = {"deepseek": "Extract (DeepSeek)", "qwen3vl": "Extract (Qwen3 VL)", "docling": "Extract (Docling)"}
    extract_label = extract_labels.get(backend, f"Extract ({backend})")
    steps = [
        "Inventory",
        extract_label,
        "Extract (MuPDF)",
        "Extract metadata",
        "Normalize text",
        "Detect boilerplate",
        "Detect content area",
        "Detect language",
        "OCR cleanup",
        "Detect footnotes",
        "Link footnote refs",
        "Footnote ref markup",
        "Detect rotation",
        "Detect figures",
        "Detect captions",
        "Export JSONL",
    ]

    progress = tqdm(steps, desc=pdf_path.name, unit="step")

    # Step 1: Inventory
    progress.set_postfix_str("Inventory")
    document = run_inventory(config)
    doc_id = document.doc_id
    progress.update(1)

    # Step 2: Text extraction (Docling, DeepSeek, or Qwen3 VL)
    if config.extraction_backend == "deepseek":
        progress.set_postfix_str("DeepSeek-OCR")
        blocks_by_page = extract_with_deepseek(config, doc_id)
    elif config.extraction_backend == "qwen3vl":
        progress.set_postfix_str("Qwen3 VL")
        blocks_by_page = extract_with_qwen3vl(config, doc_id)
    else:
        progress.set_postfix_str("Docling")
        blocks_by_page = extract_with_docling(config, doc_id)
    progress.update(1)

    # Step 3: MuPDF extraction (also extracts PDF metadata)
    progress.set_postfix_str("MuPDF")
    mupdf_data = extract_with_mupdf(config)
    pdf_metadata = extract_pdf_metadata(config)
    progress.update(1)

    # Build page records from MuPDF data
    page_records: list[PageRecord] = []
    page_heights: dict[int, float] = {}
    for page_num, pd in sorted(mupdf_data.items()):
        page_heights[page_num] = pd.height
        block_hashes = [text_hash(b.text_raw) for b in blocks_by_page.get(page_num, [])]
        p_hash = page_content_hash(block_hashes) if block_hashes else None

        page_records.append(
            PageRecord(
                doc_id=doc_id,
                page_index_0=page_num - 1,
                page_num_1=page_num,
                width=pd.width,
                height=pd.height,
                page_rotation=pd.rotation,
                page_hash=p_hash,
                text_char_count=sum(len(b.text_raw) for b in blocks_by_page.get(page_num, [])),
                image_count=len(pd.images),
            )
        )

    existing_pages = {pr.page_num_1 for pr in page_records}
    for page_num in blocks_by_page:
        if page_num not in existing_pages:
            page_heights[page_num] = 800.0
            page_records.append(
                PageRecord(
                    doc_id=doc_id,
                    page_index_0=page_num - 1,
                    page_num_1=page_num,
                    width=612.0,
                    height=792.0,
                    text_char_count=sum(len(b.text_raw) for b in blocks_by_page.get(page_num, [])),
                )
            )
    page_records.sort(key=lambda p: p.page_num_1)

    # Step 4: Extract metadata
    progress.set_postfix_str("Metadata")
    document = extract_metadata(document, blocks_by_page, pdf_metadata)
    progress.update(1)

    # Step 4b: Apply page_offset to page numbers (adjusts for JSTOR/MUSE cover pages)
    if document.page_offset and document.page_offset != 0:
        offset = document.page_offset
        logger.info("Applying page_offset=%d to all page numbers", offset)
        # Re-key blocks_by_page
        blocks_by_page = {pn + offset: blocks for pn, blocks in blocks_by_page.items()}
        for blocks in blocks_by_page.values():
            for block in blocks:
                block.page_num_1 = block.page_num_1 + offset
        # Re-key page_heights
        page_heights = {pn + offset: h for pn, h in page_heights.items()}
        # Adjust page_records
        for pr in page_records:
            pr.page_num_1 = pr.page_num_1 + offset

    # Step 5: Normalize text
    progress.set_postfix_str("Normalize")
    blocks_by_page = normalize_blocks(blocks_by_page)
    progress.update(1)

    # Step 6: Boilerplate detection
    progress.set_postfix_str("Boilerplate")
    blocks_by_page, removed_blocks = detect_boilerplate(blocks_by_page, page_heights, config)
    progress.update(1)

    # Step 7: Content area detection
    progress.set_postfix_str("Content area")
    page_records = detect_content_area(blocks_by_page, page_records)
    progress.update(1)

    # Step 8: Language detection
    progress.set_postfix_str("Language")
    blocks_by_page = detect_languages(blocks_by_page, config)
    progress.update(1)

    # Step 9: OCR cleanup
    progress.set_postfix_str("OCR cleanup")
    blocks_by_page = ocr_cleanup_blocks(blocks_by_page)
    progress.update(1)

    # Step 10: Footnote detection
    progress.set_postfix_str("Footnotes")
    blocks_by_page, footnotes = detect_footnotes(blocks_by_page, page_heights, config, doc_id)
    for page_num in blocks_by_page:
        blocks_by_page[page_num] = [b for b in blocks_by_page[page_num] if b.block_type != "footnote"]
    progress.update(1)

    # Step 11: Link footnote references
    progress.set_postfix_str("Footnote refs")
    footnote_refs = link_footnote_refs(blocks_by_page, footnotes, mupdf_data, doc_id)
    progress.update(1)

    # Step 12: Footnote ref markup
    progress.set_postfix_str("Ref markup")
    blocks_by_page = apply_ref_markup(blocks_by_page, footnote_refs)
    progress.update(1)

    # Step 13: Rotation detection
    progress.set_postfix_str("Rotation")
    page_records, page_rotations = detect_rotation(mupdf_data, page_records)
    progress.update(1)

    # Step 14: Figure detection
    progress.set_postfix_str("Figures")
    figures = detect_figures(mupdf_data, config, doc_id, page_rotations)
    progress.update(1)

    # Step 15: Caption detection
    progress.set_postfix_str("Captions")
    figures, plates = detect_captions(figures, blocks_by_page, doc_id)
    progress.update(1)

    # Step 16: Export
    progress.set_postfix_str("Export")
    export_all(
        out_dir=config.out_dir,
        doc_id=doc_id,
        document=document,
        pages=page_records,
        blocks_by_page=blocks_by_page,
        removed_blocks=removed_blocks,
        footnotes=footnotes,
        footnote_refs=footnote_refs,
        figures=figures,
        plates=plates,
    )
    progress.update(1)
    progress.close()

    return doc_id


@cli.command()
@click.option("--pdf", "pdf_path", required=True, type=click.Path(exists=True, path_type=Path), help="Path to PDF file")
@click.option("--out-dir", default="data", type=click.Path(path_type=Path), help="Output directory")
@click.option("--max-pages", default=None, type=int, help="Maximum pages to process")
@click.option("--page-range", default=None, type=str, help="Page range (e.g. '1-10' or '5,10,15')")
@click.option("--force", is_flag=True, help="Force re-processing even if output exists")
@click.option("--backend", default=None, type=click.Choice(["docling", "deepseek", "qwen3vl"]), help="Extraction backend (auto-detects if not set)")
def run(pdf_path: Path, out_dir: Path, max_pages: int | None, page_range: str | None, force: bool, backend: str | None) -> None:
    """Run the full document processing pipeline on a single PDF."""
    doc_id = _run_pipeline(pdf_path, out_dir, max_pages, page_range, force, backend)
    console.print(f"\n[bold green]Done![/] {doc_id}")


def _run_one(args: tuple) -> tuple[str, str | None]:
    """Worker function for ProcessPoolExecutor. Returns (filename, error_or_None)."""
    pdf_path, out_dir, force, backend = args
    try:
        _run_pipeline(pdf_path, out_dir, None, None, force, backend)
        return (pdf_path.name, None)
    except Exception as e:
        return (pdf_path.name, str(e))


@cli.command("run-all")
@click.option("--docs-dir", default="docs", type=click.Path(exists=True, path_type=Path), help="Directory containing PDFs")
@click.option("--out-dir", default="data", type=click.Path(path_type=Path), help="Output directory")
@click.option("--force", is_flag=True, help="Force re-processing even if output exists")
@click.option("--workers", default=None, type=int, help="Number of parallel workers (default: CPU count)")
@click.option("--backend", default=None, type=click.Choice(["docling", "deepseek", "qwen3vl"]), help="Extraction backend (auto-detects if not set)")
def run_all(docs_dir: Path, out_dir: Path, force: bool, workers: int | None, backend: str | None) -> None:
    """Process all PDFs in a directory in parallel."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if workers is None:
        workers = os.cpu_count() or 4

    pdfs = sorted(docs_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {docs_dir}[/]")
        return

    console.print(f"Found [bold]{len(pdfs)}[/] PDFs, processing with {workers} workers\n")

    args_list = [(pdf, out_dir, force, backend) for pdf in pdfs]

    succeeded = 0
    failed: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, args): args[0].name for args in args_list}
        for future in as_completed(futures):
            name, error = future.result()
            if error:
                failed.append((name, error))
                console.print(f"[red]FAIL[/] {name}: {error}")
            else:
                succeeded += 1
                console.print(f"[green]OK[/]   {name}")

    console.print(f"\n[bold]Results:[/] {succeeded} succeeded, {len(failed)} failed out of {len(pdfs)}")
    for name, error in failed:
        console.print(f"  [red]FAIL[/] {name}: {error}")


@cli.command()
@click.option("--doc-id", required=True, help="Document ID")
@click.option("--pages", default=None, help="Comma-separated page numbers to include")
@click.option("--out-dir", default="data", type=click.Path(path_type=Path), help="Data directory")
@click.option("--output", default=None, type=click.Path(path_type=Path), help="Output HTML path")
def report(doc_id: str, pages: str | None, out_dir: Path, output: Path | None) -> None:
    """Generate an HTML debug report for a processed document."""
    from ras_docproc.pipeline.report_html import generate_report

    doc_dir = out_dir / "out" / doc_id
    if not doc_dir.exists():
        console.print(f"[red]Error:[/] Document directory not found: {doc_dir}")
        sys.exit(1)

    page_list = None
    if pages:
        page_list = [int(p.strip()) for p in pages.split(",")]

    if output is None:
        output = Path("reports") / f"{doc_id}_report.html"

    generate_report(doc_dir, output, page_filter=page_list)
    console.print(f"[bold green]Report generated:[/] {output}")


@cli.command(name="debug-page")
@click.option("--pdf", "pdf_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--page", required=True, type=int, help="Page number (1-based)")
@click.option("--out", default="debug_page.png", type=click.Path(path_type=Path))
@click.option("--dpi", default=150, type=int)
def debug_page(pdf_path: Path, page: int, out: Path, dpi: int) -> None:
    """Render a single page with overlays for debugging."""
    import fitz

    doc = fitz.open(str(pdf_path))
    if page < 1 or page > len(doc):
        console.print(f"[red]Error:[/] Page {page} out of range (1-{len(doc)})")
        sys.exit(1)

    p = doc[page - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = p.get_pixmap(matrix=mat)
    pix.save(str(out))
    doc.close()
    console.print(f"[bold green]Rendered:[/] {out} ({pix.width}x{pix.height}px)")


if __name__ == "__main__":
    cli()
