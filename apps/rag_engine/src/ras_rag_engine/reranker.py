"""Reranking via AWS Bedrock."""

from __future__ import annotations

import logging

from .config import RAGConfig
from .providers import get_rerank_provider
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)


def _doc_text(chunk: RetrievedChunk) -> str:
    """Prepend document title/author metadata to chunk text for reranking context."""
    parts = []
    if chunk.author:
        parts.append(f"Author: {chunk.author}")
    if chunk.title:
        parts.append(f"Document: {chunk.title}")
    parts.append(chunk.text)
    return "\n".join(parts)


def rerank(query: str, chunks: list[RetrievedChunk], config: RAGConfig, top_k: int) -> list[RetrievedChunk]:
    """Rerank chunks using the configured backend and return top_k by relevance."""
    if not chunks:
        return chunks

    provider = get_rerank_provider(config)
    documents = [_doc_text(c) for c in chunks]
    ranked = provider.rerank(query, documents, top_k)

    result = []
    for idx, score in ranked:
        chunk = chunks[idx]
        chunk.score = score
        log.debug("rerank: %.4f  %s...", score, chunk.text[:80])
        result.append(chunk)

    return result
