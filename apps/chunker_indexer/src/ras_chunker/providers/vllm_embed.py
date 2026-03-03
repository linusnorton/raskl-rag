"""vLLM embedding provider — OpenAI-compatible embeddings API."""

from __future__ import annotations

import math

import httpx

from . import EmbedProvider


def _truncate_and_normalize(embedding: list[float], dimensions: int) -> list[float]:
    vec = embedding[:dimensions]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class VLLMEmbedProvider(EmbedProvider):
    def __init__(self, base_url: str, model: str, batch_size: int, dimensions: int, task_prefix: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.task_prefix = task_prefix

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self.task_prefix}{t}" for t in texts]
        all_embeddings: list[list[float]] = [[] for _ in texts]

        with httpx.Client(timeout=120.0) as client:
            for batch_start in range(0, len(prefixed), self.batch_size):
                batch = prefixed[batch_start : batch_start + self.batch_size]
                resp = client.post(
                    f"{self.base_url}/v1/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                for item in sorted(data, key=lambda x: x["index"]):
                    embedding = item["embedding"]
                    if len(embedding) > self.dimensions:
                        embedding = _truncate_and_normalize(embedding, self.dimensions)
                    all_embeddings[batch_start + item["index"]] = embedding

        return all_embeddings
