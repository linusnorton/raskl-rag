"""Pipeline step: LLM-based metadata extraction from first N pages of OCR text."""

from __future__ import annotations

import json
import logging
import re

from ras_docproc.config import PipelineConfig
from ras_docproc.pipeline.classify_doctype import VALID_DOCUMENT_TYPES
from ras_docproc.schema import DocumentRecord, MetadataSource, TextBlockRecord

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
You are a metadata extraction engine for a historical document archive (primarily JMBRAS — \
Journal of the Malaysian Branch of the Royal Asiatic Society).

Given the text from the opening pages of a document, extract structured bibliographic metadata. \
Return ONLY a JSON object (no markdown fencing) with these fields:

{
  "title": "full document title",
  "author": "author name(s), semicolon-separated if multiple",
  "editor": "editor name(s) if applicable, else null",
  "year": 1234,
  "publication": "journal or publisher name",
  "volume": "volume number or null",
  "issue": "issue/part number or null",
  "abstract": "abstract text if present, else null",
  "keywords": ["keyword1", "keyword2"],
  "language": "primary language code (en, ms, zh, ar, etc.)",
  "isbn": "ISBN if found, else null",
  "issn": "ISSN if found, else null",
  "series": "series name if applicable (e.g. 'JMBRAS Monograph No. 12'), else null",
  "description": "2-3 sentence summary of what the document is about",
  "doi": "DOI if found, else null",
  "document_type": "journal_article | front_matter | obituary | editors_note | annual_report | agm_minutes | biographical_notes | secondary_source | primary_source | mbras_monograph | mbras_reprint | index | illustration"
}

Rules:
- Extract only what is stated or clearly implied in the text. Do not fabricate.
- For title: prefer the main document title, not the journal name.
- For author: use the form as written (e.g. "Abdullah, Munshi" or "Frank A. Swettenham").
- For year: use the publication year from the journal citation (volume/issue info), not a historical date mentioned in the document content. For example, if the journal says "Vol. 47, No. 2 (1974)" and the article discusses events in 1500, the year is 1974.
- For description: write a brief factual summary based on the content you see.
- For keywords: extract 3-8 terms covering the subject, region, period, and key topics.
- Set null for any field you cannot determine from the text."""


def enrich_metadata_llm(
    document: DocumentRecord,
    blocks_by_page: dict[int, list[TextBlockRecord]],
    config: PipelineConfig,
) -> DocumentRecord:
    """Extract metadata from the first 10 pages of OCR text using an LLM.

    Sends concatenated text from up to 10 substantive pages to the Bedrock LLM
    for structured metadata extraction. Only fills in fields that are currently None/empty.

    Mutates and returns the DocumentRecord.
    """
    # Collect text from up to 10 substantive pages
    _MIN_PAGE_TEXT = 50
    collected: list[str] = []
    collected_len = 0
    for page_num in sorted(blocks_by_page.keys()):
        page_text = "\n".join((b.text_clean or b.text_raw) for b in blocks_by_page[page_num]).strip()
        if len(page_text) < _MIN_PAGE_TEXT:
            continue
        collected.append(page_text)
        collected_len += len(page_text)
        if len(collected) >= 10:
            break

    if not collected:
        logger.warning("No substantive pages found for LLM metadata extraction")
        return document

    opening_text = "\n\n---\n\n".join(collected)[:8000]
    user_message = f"DOCUMENT TEXT (first pages):\n\n{opening_text}"

    try:
        if config.llm_provider == "model_studio":
            from openai import OpenAI

            ms_client = OpenAI(api_key=config.model_studio_api_key, base_url=config.model_studio_base_url)
            resp = ms_client.chat.completions.create(
                model=config.model_studio_ocr_model_id,
                messages=[
                    {"role": "system", "content": _EXTRACT_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            raw_text = resp.choices[0].message.content or ""
        else:
            import boto3
            from botocore.config import Config as BotoConfig

            bedrock_config = BotoConfig(read_timeout=120, retries={"max_attempts": 2})
            bedrock_client = boto3.client("bedrock-runtime", region_name=config.bedrock_region, config=bedrock_config)

            resp = bedrock_client.converse(
                modelId=config.bedrock_ocr_model_id,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": _EXTRACT_PROMPT}],
                inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
            )
            raw_text = resp["output"]["message"]["content"][0]["text"]

        # Strip <think>...</think> blocks (Qwen3 reasoning)
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # Strip markdown code fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        result = json.loads(raw_text)
        _apply_llm_result(document, result)
        logger.info("LLM metadata extraction: populated fields from first %d pages", len(collected))

    except Exception:
        logger.warning("LLM metadata extraction failed", exc_info=True)

    return document


def _apply_llm_result(document: DocumentRecord, result: dict) -> None:
    """Apply LLM extraction results to document, only filling empty fields."""
    field_map = {
        "title": "title",
        "author": "author",
        "editor": "editor",
        "year": "year",
        "publication": "publication",
        "volume": "volume",
        "issue": "issue",
        "abstract": "abstract",
        "language": "language",
        "isbn": "isbn",
        "issn": "issn",
        "series": "series",
        "description": "description",
        "doi": "doi",
    }

    for json_key, field_name in field_map.items():
        value = result.get(json_key)
        if value is None:
            continue

        current = getattr(document, field_name, None)

        # For year, convert to int
        if field_name == "year":
            try:
                value = int(value)
            except (ValueError, TypeError):
                continue

        # Only fill if current value is None/empty
        if current is None or current == "":
            setattr(document, field_name, value)
            document.metadata_sources.append(
                MetadataSource(field=field_name, source="llm_extraction", confidence=0.8, raw_value=str(value))
            )

    # Keywords always merge (additive)
    llm_keywords = result.get("keywords", [])
    if isinstance(llm_keywords, list) and llm_keywords:
        existing = set(document.keywords)
        for kw in llm_keywords:
            if isinstance(kw, str) and kw.strip() and kw.strip() not in existing:
                document.keywords.append(kw.strip())
                existing.add(kw.strip())
        if llm_keywords:
            document.metadata_sources.append(
                MetadataSource(
                    field="keywords", source="llm_extraction", confidence=0.8, raw_value=", ".join(llm_keywords)
                )
            )

    # Document type from LLM (if not already set by classify_doctype stage)
    doc_type = result.get("document_type", "")
    if doc_type in VALID_DOCUMENT_TYPES and not document.document_type:
        document.document_type = doc_type
        document.metadata_sources.append(
            MetadataSource(field="document_type", source="llm_extraction", confidence=0.7, raw_value=doc_type)
        )
