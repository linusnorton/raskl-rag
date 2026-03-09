"""Pipeline step: classify document type using LLM analysis of first pages + metadata."""

from __future__ import annotations

import json
import logging

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import DocumentRecord, TextBlockRecord

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
You are a document classifier for a historical archive. Given the metadata and opening text of a document, classify it into one of these types:

- **primary_source**: A document created during the historical period it describes — diaries, journals, letters, government records, contemporary accounts, translations of historical texts.
- **journal_article**: A peer-reviewed academic paper published in a scholarly journal, with modern scholarly apparatus (abstract, citations, bibliography).

Signals for journal_article: presence of an abstract, numbered footnotes citing other works, a bibliography/references section, academic author attribution with year, JSTOR/Project MUSE metadata, analytical language.

Signals for primary_source: diary entries with dates, first-person accounts, correspondence, government dispatches, translations of historical documents, no modern scholarly apparatus.

Respond with ONLY a JSON object (no markdown fencing):
{"document_type": "journal_article" or "primary_source", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


def classify_document_type(
    document: DocumentRecord,
    blocks_by_page: dict[int, list[TextBlockRecord]],
    config: PipelineConfig,
) -> DocumentRecord:
    """Classify the document type using LLM analysis of metadata + first 2 pages.

    Mutates and returns the DocumentRecord with populated document_type field.
    """
    # Build context from metadata + first 2 pages of text
    meta_parts = []
    if document.title:
        meta_parts.append(f"Title: {document.title}")
    if document.author:
        meta_parts.append(f"Author: {document.author}")
    if document.year:
        meta_parts.append(f"Year: {document.year}")
    if document.publication:
        meta_parts.append(f"Publication: {document.publication}")
    if document.journal_ref:
        meta_parts.append(f"Journal ref: {document.journal_ref}")
    if document.doi:
        meta_parts.append(f"DOI: {document.doi}")
    meta_parts.append(f"Filename: {document.source_filename}")

    meta_text = "\n".join(meta_parts)

    # Collect text from first 2 pages
    page_texts = []
    for page_num in sorted(blocks_by_page.keys())[:2]:
        for block in blocks_by_page[page_num]:
            text = block.text_clean or block.text_raw
            page_texts.append(text)

    opening_text = "\n\n".join(page_texts)[:3000]  # Cap at ~3000 chars

    user_message = f"METADATA:\n{meta_text}\n\nOPENING TEXT:\n{opening_text}"

    try:
        import boto3
        from botocore.config import Config as BotoConfig

        bedrock_config = BotoConfig(read_timeout=60, retries={"max_attempts": 2})
        client = boto3.client("bedrock-runtime", region_name=config.bedrock_region, config=bedrock_config)

        response = client.converse(
            modelId=config.bedrock_ocr_model_id,
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            system=[{"text": _CLASSIFY_PROMPT}],
            inferenceConfig={"maxTokens": 256, "temperature": 0.0},
        )

        raw_text = response["output"]["message"]["content"][0]["text"]

        # Strip <think>...</think> blocks (Qwen3 reasoning)
        import re

        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        result = json.loads(raw_text)
        doc_type = result.get("document_type", "").strip().lower()
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")

        if doc_type in ("journal_article", "primary_source"):
            document.document_type = doc_type
            logger.info("Document classified as '%s' (confidence=%.2f): %s", doc_type, confidence, reasoning)
        else:
            logger.warning("LLM returned unknown document_type '%s', leaving as None", doc_type)

    except Exception:
        logger.warning("Document type classification failed, leaving as None", exc_info=True)

    return document
