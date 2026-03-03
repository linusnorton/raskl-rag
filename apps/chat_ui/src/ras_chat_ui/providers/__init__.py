"""Provider factories for LLM, embedding, and reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EmbedProvider, LLMProvider, RerankProvider

if TYPE_CHECKING:
    from ..config import ChatConfig


def get_llm_provider(config: ChatConfig) -> LLMProvider:
    if config.llm_provider == "bedrock":
        from .bedrock_llm import BedrockLLMProvider

        return BedrockLLMProvider(
            region=config.bedrock_region,
            model_id=config.bedrock_chat_model_id,
        )
    else:
        from .vllm_llm import VLLMLLMProvider

        return VLLMLLMProvider(
            base_url=config.llm_base_url,
            model=config.llm_model,
        )


def get_embed_provider(config: ChatConfig) -> EmbedProvider:
    if config.embed_provider == "bedrock":
        from .bedrock_embed import BedrockEmbedProvider

        return BedrockEmbedProvider(
            region=config.bedrock_region,
            model_id=config.bedrock_embed_model_id,
            dimensions=config.embed_dimensions,
            task_prefix=config.embed_task_prefix,
            input_type="search_query",
        )
    else:
        from .local_embed import LocalEmbedProvider

        return LocalEmbedProvider(
            model_path=config.embed_model,
            device=config.embed_device,
            dimensions=config.embed_dimensions,
            task_prefix=config.embed_task_prefix,
        )


def get_rerank_provider(config: ChatConfig) -> RerankProvider:
    if config.rerank_provider == "bedrock":
        from .bedrock_rerank import BedrockRerankProvider

        return BedrockRerankProvider(
            region=config.bedrock_region,
            model_id=config.bedrock_rerank_model_id,
        )
    elif config.rerank_provider == "cross-encoder":
        from .local_rerank import CrossEncoderRerankProvider

        return CrossEncoderRerankProvider(
            model_path=config.rerank_model,
            device=config.rerank_device,
        )
    else:
        from .local_rerank import Qwen3RerankProvider

        return Qwen3RerankProvider(
            model_path=config.rerank_model,
            device=config.rerank_device,
            instruction=config.rerank_instruction,
        )
