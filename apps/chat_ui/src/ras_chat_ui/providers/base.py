"""Abstract base classes for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class LLMProvider(ABC):
    """Abstract LLM provider for chat completions."""

    @abstractmethod
    def count_tokens(self, messages: list[dict], *, tools: list[dict] | None = None) -> int:
        """Estimate or count the number of input tokens."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> dict:
        """Non-streaming chat completion. Returns parsed response dict with keys:
        - tool_calls: list[dict] | None  (each has id, function.name, function.arguments)
        - reasoning: str | None
        - content: str | None
        """

    @abstractmethod
    def chat_completion_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Generator[dict[str, str], None, None]:
        """Streaming chat completion. Yields dicts with optional keys:
        - reasoning: str (thinking token)
        - content: str (output token)
        Tools must be passed when message history contains tool_calls/tool results.
        """


class EmbedProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""


class RerankProvider(ABC):
    """Abstract reranking provider."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Rerank documents against a query. Returns list of (original_index, score) sorted by score descending."""
