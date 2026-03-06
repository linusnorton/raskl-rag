"""AWS Bedrock embedding provider — supports Cohere Embed and Amazon Titan Text Embeddings."""

from __future__ import annotations

import json
import logging

from . import EmbedProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(region: str):
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-runtime", region_name=region)
    return _client


def _is_titan(model_id: str) -> bool:
    return "titan-embed" in model_id


class BedrockEmbedProvider(EmbedProvider):
    def __init__(self, region: str, model_id: str, dimensions: int, task_prefix: str):
        self.region = region
        self.model_id = model_id
        self.dimensions = dimensions
        self.task_prefix = task_prefix

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = _get_client(self.region)
        prefixed = [f"{self.task_prefix}{t}" for t in texts]

        if _is_titan(self.model_id):
            return self._embed_titan(client, prefixed)
        return self._embed_cohere(client, prefixed)

    def _embed_titan(self, client, texts: list[str]) -> list[list[float]]:
        """Amazon Titan Text Embeddings — concurrent calls via ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        def _embed_one(text: str) -> list[float]:
            import boto3

            thread_client = boto3.client("bedrock-runtime", region_name=self.region)
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True,
            })
            resp = thread_client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            return result["embedding"]

        with ThreadPoolExecutor(max_workers=10) as pool:
            all_embeddings = list(pool.map(_embed_one, texts))
        return all_embeddings

    def _embed_cohere(self, client, texts: list[str]) -> list[list[float]]:
        """Cohere Embed — up to 96 texts per batch."""
        all_embeddings: list[list[float]] = []
        batch_size = 96

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            body = json.dumps({
                "texts": batch,
                "input_type": "search_document",
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
