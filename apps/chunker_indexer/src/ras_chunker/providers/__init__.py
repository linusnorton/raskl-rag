"""Embedding provider factory for chunker/indexer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ChunkerConfig


class EmbedProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""


def get_embed_provider(config: ChunkerConfig) -> EmbedProvider:
    if config.embed_provider == "bedrock":
        from .bedrock_embed import BedrockEmbedProvider

        return BedrockEmbedProvider(
            region=config.bedrock_region,
            model_id=config.bedrock_embed_model_id,
            dimensions=config.embed_dimensions,
            task_prefix=config.embed_task_prefix,
        )
    else:
        from .vllm_embed import VLLMEmbedProvider

        return VLLMEmbedProvider(
            base_url=config.embed_base_url,
            model=config.embed_model,
            batch_size=config.embed_batch_size,
            dimensions=config.embed_dimensions,
            task_prefix=config.embed_task_prefix,
        )
