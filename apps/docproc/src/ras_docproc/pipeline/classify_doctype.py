"""Pipeline step: classify document type using LLM analysis of first pages + metadata."""

from __future__ import annotations

import json
import logging

from ras_docproc.config import PipelineConfig
from ras_docproc.schema import DocumentRecord, TextBlockRecord

logger = logging.getLogger(__name__)

VALID_DOCUMENT_TYPES = {
    "journal_article",
    "front_matter",
    "obituary",
    "editors_note",
    "annual_report",
    "agm_minutes",
    "biographical_notes",
    "secondary_source",
    "primary_source",
    "mbras_monograph",
    "mbras_reprint",
    "index",
    "illustration",
}

_CLASSIFY_PROMPT = """\
You are a document classifier for a historical archive (primarily JMBRAS — Journal of the Malaysian Branch of the Royal Asiatic Society). Given the metadata and opening text of a document, classify it into exactly one of these types:

- **journal_article**: A peer-reviewed academic paper with formal scholarly apparatus — abstract, numbered citations, bibliography/references section, academic author attribution. Published in a scholarly journal.
- **secondary_source**: A longer historical work or essay that analyses past events but lacks the formal apparatus of a journal article (no abstract, no bibliography section). Includes historical surveys, retrospective essays, and book chapters.
- **primary_source**: A document created during the historical period it describes — diaries, journals, letters, government records, dispatches, contemporary accounts, translations of historical texts. Classify based on content, not publication channel (e.g. a diary published via JSTOR is still a primary source).
- **mbras_monograph**: An original book-length work published by MBRAS/JMBRAS as a standalone monograph or monograph series entry (e.g. "MBRAS Monograph No. 12"). Typically has its own ISBN or series number.
- **mbras_reprint**: An older out-of-print book or text reprinted/republished by MBRAS. Often has a modern preface or introduction followed by a much older original text.
- **front_matter**: Non-article content at the start of a journal issue — table of contents, member lists, patron lists, council lists, office-bearer lists, institutional information.
- **obituary**: A memorial notice or tribute for a deceased person, typically naming the subject and summarising their life and contributions.
- **editors_note**: An editor's foreword, preface, introduction, or editorial note for a journal issue or volume. Written by the editor rather than a contributing author. Filenames often contain "EditorsNote".
- **illustration**: A standalone page or PDF of illustrations, photographs, plates, or maps extracted separately (e.g. by JSTOR). Filenames often match "Illustration-{year}.pdf". Contains primarily images with minimal or no body text.
- **annual_report**: A yearly report of the society's activities, finances, membership, publications, and events.
- **agm_minutes**: Minutes or proceedings of the Annual General Meeting of the society, including motions, votes, and resolutions.
- **biographical_notes**: Short biographical sketches or notes about contributors, members, or historical figures. Shorter and less formal than an obituary.
- **index**: A subject index, author index, or cumulative index for one or more journal volumes.

Key disambiguation:
- journal_article vs secondary_source: journal articles have formal academic apparatus (abstract, citations, bibliography); secondary sources are analytical but lack this formal structure.
- mbras_monograph vs mbras_reprint: monographs are original works; reprints are older texts republished by MBRAS.
- editors_note includes forewords and prefaces written by editors.
- front_matter covers administrative/institutional content, not articles.
- illustration is for standalone image pages/PDFs (often JSTOR extracts), not figures embedded within articles.

Respond with ONLY a JSON object (no markdown fencing):
{"document_type": "<one of the types above>", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


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
        page_text = "\n".join((b.text_clean or b.text_raw) for b in blocks_by_page[page_num]).strip()
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

        if doc_type in VALID_DOCUMENT_TYPES:
            document.document_type = doc_type
            logger.info("Document classified as '%s' (confidence=%.2f): %s", doc_type, confidence, reasoning)
        else:
            logger.warning("LLM returned unknown document_type '%s', leaving as None", doc_type)

    except Exception:
        logger.warning("Document type classification failed, leaving as None", exc_info=True)

    return document
