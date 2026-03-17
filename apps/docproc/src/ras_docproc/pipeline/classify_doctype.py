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

Signals for primary_source: diary entries with dates, first-person accounts, correspondence, government dispatches, translations of historical documents, no modern scholarly apparatus. Note: a diary or journal published by a scholarly society (e.g. via JSTOR) is still a primary source, not a journal article — classify based on the content, not the publication channel.

Respond with ONLY a JSON object (no markdown fencing):
{"document_type": "journal_article" or "primary_source", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


def classify_document_type(
    document: DocumentRecord,
    blocks_by_page: dict[int, list[TextBlockRecord]],
    config: PipelineConfig,
) -> DocumentRecord:
    """Classify the document type using LLM analysis of metadata + opening text.

    Collects the first ~3000 characters of actual text content, skipping
    pages that are blank, title sheets, or image-only (e.g. JSTOR cover pages).
    This avoids misclassification when early pages lack substantive content.

    Mutates and returns the DocumentRecord with populated document_type field.
    """
    # Build context from metadata
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

    # Collect text across pages until we have enough substantive content.
    # Skip pages with very little text (title sheets, blanks, images).
    _MIN_PAGE_TEXT = 50  # characters — skip pages shorter than this
    collected: list[str] = []
    collected_len = 0
    for page_num in sorted(blocks_by_page.keys()):
        page_text = "\n".join(
            (b.text_clean or b.text_raw) for b in blocks_by_page[page_num]
        ).strip()
        if len(page_text) < _MIN_PAGE_TEXT:
            continue
        collected.append(page_text)
        collected_len += len(page_text)
        if collected_len >= 3000:
            break

    opening_text = "\n\n".join(collected)[:3000]

    user_message = f"METADATA:\n{meta_text}\n\nOPENING TEXT:\n{opening_text}"

    try:
        import re

        if config.llm_provider == "model_studio":
            from openai import OpenAI

            ms_client = OpenAI(api_key=config.model_studio_api_key, base_url=config.model_studio_base_url)
            resp = ms_client.chat.completions.create(
                model=config.model_studio_ocr_model_id,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw_text = resp.choices[0].message.content or ""
        else:
            import boto3
            from botocore.config import Config as BotoConfig

            bedrock_config = BotoConfig(read_timeout=60, retries={"max_attempts": 2})
            bedrock_client = boto3.client("bedrock-runtime", region_name=config.bedrock_region, config=bedrock_config)

            resp = bedrock_client.converse(
                modelId=config.bedrock_ocr_model_id,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": _CLASSIFY_PROMPT}],
                inferenceConfig={"maxTokens": 256, "temperature": 0.0},
            )
            raw_text = resp["output"]["message"]["content"][0]["text"]

        # Strip <think>...</think> blocks (Qwen3 reasoning)
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
