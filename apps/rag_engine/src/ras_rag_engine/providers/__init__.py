"""Provider factories for LLM, embedding, and reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EmbedProvider, LLMProvider, RerankProvider

if TYPE_CHECKING:
    from ..config import RAGConfig


def get_llm_provider(config: RAGConfig) -> LLMProvider:
    if config.llm_provider == "model_studio":
        from .model_studio_llm import ModelStudioLLMProvider

        return ModelStudioLLMProvider(
            api_key=config.model_studio_api_key,
            base_url=config.model_studio_base_url,
            model_id=config.model_studio_chat_model_id,
            thinking_budget=config.llm_thinking_budget,
        )

    from .bedrock_llm import BedrockLLMProvider

    return BedrockLLMProvider(
        region=config.bedrock_region,
        model_id=config.bedrock_chat_model_id,
        thinking_budget=config.llm_thinking_budget,
    )


def get_embed_provider(config: RAGConfig) -> EmbedProvider:
    if config.llm_provider == "model_studio":
        from .model_studio_embed import ModelStudioEmbedProvider

        return ModelStudioEmbedProvider(
            api_key=config.model_studio_api_key,
            base_url=config.model_studio_base_url,
            model_id=config.model_studio_embed_model_id,
            dimensions=config.embed_dimensions,
            task_prefix=config.embed_task_prefix,
        )

    from .bedrock_embed import BedrockEmbedProvider

    return BedrockEmbedProvider(
        region=config.bedrock_embed_region,
        model_id=config.bedrock_embed_model_id,
        dimensions=config.embed_dimensions,
        task_prefix=config.embed_task_prefix,
        input_type="search_query",
    )


def get_rerank_provider(config: RAGConfig) -> RerankProvider:
    if config.llm_provider == "model_studio":
        from .model_studio_rerank import ModelStudioRerankProvider

        return ModelStudioRerankProvider(
            api_key=config.model_studio_api_key,
            base_url=config.model_studio_base_url,
            model_id=config.model_studio_rerank_model_id,
            query_prefix=config.rerank_instruction + ": " if config.rerank_instruction else "",
        )

    from .bedrock_rerank import BedrockRerankProvider

    return BedrockRerankProvider(
        region=config.bedrock_rerank_region,
        model_id=config.bedrock_rerank_model_id,
        query_prefix=config.rerank_instruction + ": " if config.rerank_instruction else "",
    )
