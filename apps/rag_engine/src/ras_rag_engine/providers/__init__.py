"""Provider factories for LLM, embedding, and reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EmbedProvider, LLMProvider, RerankProvider

if TYPE_CHECKING:
    from ..config import RAGConfig


def get_llm_provider(config: RAGConfig) -> LLMProvider:
    from .bedrock_llm import BedrockLLMProvider

    return BedrockLLMProvider(
        region=config.bedrock_region,
        model_id=config.bedrock_chat_model_id,
        thinking_budget=config.llm_thinking_budget,
    )


def get_embed_provider(config: RAGConfig) -> EmbedProvider:
    from .bedrock_embed import BedrockEmbedProvider

    return BedrockEmbedProvider(
        region=config.bedrock_region,
        model_id=config.bedrock_embed_model_id,
        dimensions=config.embed_dimensions,
        task_prefix=config.embed_task_prefix,
        input_type="search_query",
    )


def get_rerank_provider(config: RAGConfig) -> RerankProvider:
    from .bedrock_rerank import BedrockRerankProvider

    return BedrockRerankProvider(
        region=config.bedrock_rerank_region,
        model_id=config.bedrock_rerank_model_id,
        query_prefix=config.rerank_instruction + ": " if config.rerank_instruction else "",
    )
