"""Alibaba Model Studio reranking provider via DashScope API."""

from __future__ import annotations

import json
import logging
import urllib.request

from .base import RerankProvider

log = logging.getLogger(__name__)


class ModelStudioRerankProvider(RerankProvider):
    def __init__(self, api_key: str, base_url: str, model_id: str, query_prefix: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.query_prefix = query_prefix

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if self.query_prefix:
            query = f"{self.query_prefix}{query}"

        # DashScope rerank API — not OpenAI-compatible, use direct HTTP
        url = self.base_url.replace("/compatible-mode/v1", "/api/v1/services/rerank/text-rerank/text-rerank")
        payload = {
            "model": self.model_id,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_k,
                "return_documents": False,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        results = []
        for item in data.get("output", {}).get("results", []):
            results.append((item["index"], item["relevance_score"]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
