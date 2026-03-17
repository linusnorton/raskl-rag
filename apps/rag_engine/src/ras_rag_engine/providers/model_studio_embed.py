"""Alibaba Model Studio embedding provider via OpenAI-compatible API."""

from __future__ import annotations

import logging

from .base import EmbedProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(api_key: str, base_url: str):
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


class ModelStudioEmbedProvider(EmbedProvider):
    def __init__(self, api_key: str, base_url: str, model_id: str, dimensions: int, task_prefix: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.dimensions = dimensions
        self.task_prefix = task_prefix

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = _get_client(self.api_key, self.base_url)
        prefixed = [f"{self.task_prefix}{t}" for t in texts]

        # Model Studio embeddings API accepts batches up to 10 texts
        all_embeddings: list[list[float]] = []
        batch_size = 10

        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            resp = client.embeddings.create(
                model=self.model_id,
                input=batch,
                dimensions=self.dimensions,
            )
            for item in resp.data:
                all_embeddings.append(item.embedding[: self.dimensions])

        return all_embeddings
