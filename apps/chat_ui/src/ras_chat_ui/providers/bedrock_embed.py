"""AWS Bedrock embedding provider via Cohere Embed Multilingual v3."""

from __future__ import annotations

import json
import logging

from .base import EmbedProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(region: str):
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-runtime", region_name=region)
    return _client


class BedrockEmbedProvider(EmbedProvider):
    def __init__(self, region: str, model_id: str, dimensions: int, task_prefix: str, input_type: str = "search_query"):
        self.region = region
        self.model_id = model_id
        self.dimensions = dimensions
        self.task_prefix = task_prefix
        self.input_type = input_type

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = _get_client(self.region)
        prefixed = [f"{self.task_prefix}{t}" for t in texts]

        # Cohere supports up to 96 texts per batch
        all_embeddings: list[list[float]] = []
        batch_size = 96

        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            body = json.dumps({
                "texts": batch,
                "input_type": self.input_type,
                "truncate": "END",
            })

            resp = client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            embeddings = result["embeddings"]

            for emb in embeddings:
                all_embeddings.append(emb[: self.dimensions])

        return all_embeddings
