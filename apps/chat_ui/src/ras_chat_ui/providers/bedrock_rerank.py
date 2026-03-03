"""AWS Bedrock reranking provider via Cohere Rerank."""

from __future__ import annotations

import logging

from .base import RerankProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(region: str):
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-agent-runtime", region_name=region)
    return _client


class BedrockRerankProvider(RerankProvider):
    def __init__(self, region: str, model_id: str):
        self.region = region
        self.model_id = model_id

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        client = _get_client(self.region)

        text_sources = [
            {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}}}
            for doc in documents
        ]

        resp = client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=text_sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{self.model_id}",
                    },
                    "numberOfResults": top_k,
                },
            },
        )

        results = []
        for r in resp["results"]:
            results.append((r["index"], r["relevanceScore"]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
