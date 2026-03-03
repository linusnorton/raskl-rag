"""Pipeline step: export all records to JSONL files."""

from __future__ import annotations

import logging
from pathlib import Path

from ras_docproc.schema import (
    DocumentRecord,
    FigureRecord,
    FootnoteRecord,
    FootnoteRefRecord,
    PageRecord,
    PlateRecord,
    TextBlockRecord,
)
from ras_docproc.utils.io import ensure_dir, write_jsonl

logger = logging.getLogger(__name__)


def export_all(
    out_dir: Path,
    doc_id: str,
    document: DocumentRecord,
    pages: list[PageRecord],
    blocks_by_page: dict[int, list[TextBlockRecord]],
    removed_blocks: list[TextBlockRecord],
    footnotes: list[FootnoteRecord],
    footnote_refs: list[FootnoteRefRecord],
    figures: list[FigureRecord],
    plates: list[PlateRecord],
) -> Path:
    """Write all pipeline output to JSONL files under out_dir/out/{doc_id}/.

    Returns the output directory path.
    """
    doc_dir = ensure_dir(out_dir / "out" / doc_id)

    # Flatten blocks
    all_blocks: list[TextBlockRecord] = []
    for page_num in sorted(blocks_by_page.keys()):
        all_blocks.extend(blocks_by_page[page_num])

    # Write all JSONL files
    write_jsonl(doc_dir / "documents.jsonl", [document])
    logger.info("Wrote documents.jsonl (1 record)")

    write_jsonl(doc_dir / "pages.jsonl", pages)
    logger.info("Wrote pages.jsonl (%d records)", len(pages))

    write_jsonl(doc_dir / "text_blocks.jsonl", all_blocks)
    logger.info("Wrote text_blocks.jsonl (%d records)", len(all_blocks))

    write_jsonl(doc_dir / "removed_blocks.jsonl", removed_blocks)
    logger.info("Wrote removed_blocks.jsonl (%d records)", len(removed_blocks))

    write_jsonl(doc_dir / "footnotes.jsonl", footnotes)
    logger.info("Wrote footnotes.jsonl (%d records)", len(footnotes))

    write_jsonl(doc_dir / "footnote_refs.jsonl", footnote_refs)
    logger.info("Wrote footnote_refs.jsonl (%d records)", len(footnote_refs))

    write_jsonl(doc_dir / "figures.jsonl", figures)
    logger.info("Wrote figures.jsonl (%d records)", len(figures))

    if plates:
        write_jsonl(doc_dir / "plates.jsonl", plates)
        logger.info("Wrote plates.jsonl (%d records)", len(plates))

    return doc_dir
