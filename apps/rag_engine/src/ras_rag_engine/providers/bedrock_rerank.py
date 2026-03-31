"""AWS Bedrock reranking provider via Cohere Rerank 3.5."""

from __future__ import annotations

import logging

from .base import RerankProvider

log = logging.getLogger(__name__)

_clients: dict[str, object] = {}


def _get_client(region: str):
    if region not in _clients:
        import boto3

        _clients[region] = boto3.client("bedrock-agent-runtime", region_name=region)
    return _clients[region]


class BedrockRerankProvider(RerankProvider):
    def __init__(self, region: str, model_id: str, query_prefix: str = ""):
        self.region = region
        self.model_id = model_id
        self.query_prefix = query_prefix

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        client = _get_client(self.region)
        if self.query_prefix:
            query = f"{self.query_prefix}{query}"

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
                        "additionalModelRequestFields": {
                            "max_tokens_per_doc": 4096,
                        },
                    },
                    "numberOfResults": min(top_k, len(documents)),
                },
            },
        )

        results = []
        for r in resp["results"]:
            results.append((r["index"], r["relevanceScore"]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
