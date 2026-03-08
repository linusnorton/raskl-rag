"""Pipeline step: discover PDF, compute hashes, generate doc_id."""

from __future__ import annotations

import logging

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import DocumentRecord
from ras_docproc.utils.hashing import file_sha256, make_doc_id
from ras_docproc.utils.io import ensure_dir, read_jsonl

logger = logging.getLogger(__name__)


def run_inventory(config: PipelineConfig) -> DocumentRecord:
    """Discover PDF file, compute SHA256, generate doc_id.

    If output directory already exists with matching page hashes and --force is not set,
    returns the existing DocumentRecord.
    """
    pdf_path = config.pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    sha256 = file_sha256(pdf_path)
    doc_id = make_doc_id(pdf_path.name, sha256)
    out_dir = config.out_dir / "out" / doc_id

    # Check for existing output
    existing_doc_jsonl = out_dir / "documents.jsonl"
    if existing_doc_jsonl.exists() and not config.force:
        existing = read_jsonl(existing_doc_jsonl, DocumentRecord)
        if existing and existing[0].sha256_pdf == sha256:
            logger.info("Output already exists for %s (sha256 match), skipping (use --force to re-process)", doc_id)
            return existing[0]

    ensure_dir(out_dir)

    doc = DocumentRecord(
        doc_id=doc_id,
        source_path=str(pdf_path.resolve()),
        source_filename=pdf_path.name,
        sha256_pdf=sha256,
        extraction_version=config.extraction_version,
    )
    logger.info("Inventoried %s → doc_id=%s", pdf_path.name, doc_id)
    return doc
