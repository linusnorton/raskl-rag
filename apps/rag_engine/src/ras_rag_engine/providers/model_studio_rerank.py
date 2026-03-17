"""Alibaba Model Studio reranking provider via OpenAI-compatible API."""

from __future__ import annotations

import logging

from .base import RerankProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(api_key: str, base_url: str):
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


class ModelStudioRerankProvider(RerankProvider):
    def __init__(self, api_key: str, base_url: str, model_id: str, query_prefix: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.query_prefix = query_prefix

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        client = _get_client(self.api_key, self.base_url)
        if self.query_prefix:
            query = f"{self.query_prefix}{query}"

        # Model Studio rerank uses the same interface as Cohere/OpenAI rerank
        resp = client.rerank.create(
            model=self.model_id,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        results = []
        for r in resp.results:
            results.append((r.index, r.relevance_score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
