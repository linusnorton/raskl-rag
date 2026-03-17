"""Alibaba Model Studio LLM provider via OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator

from .base import LLMProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(api_key: str, base_url: str):
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _strip_thinking(text: str) -> tuple[str | None, str | None]:
    """Strip <think>...</think> blocks, returning (reasoning, content)."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = match.group(1).strip() if match else None
    content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return reasoning, content or None


class ModelStudioLLMProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model_id: str, thinking_budget: int = 0):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.thinking_budget = thinking_budget

    def count_tokens(self, messages: list[dict], *, tools: list[dict] | None = None) -> int:
        chars = sum(len(json.dumps(m)) for m in messages)
        if tools:
            chars += len(json.dumps(tools))
        return int(chars / 3.5) + 100

    def chat_completion(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> dict:
        client = _get_client(self.api_key, self.base_url)

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        reasoning = None
        content = msg.content
        if content:
            reasoning, content = _strip_thinking(content)

        return {
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "content": content,
        }

    def chat_completion_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Generator[dict[str, str], None, None]:
        client = _get_client(self.api_key, self.base_url)

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        in_thinking = False
        for chunk in client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta or not delta.content:
                continue
            text = delta.content

            # Detect <think> tags in stream and route to reasoning vs content
            if "<think>" in text:
                in_thinking = True
                text = text.split("<think>", 1)[1]
            if "</think>" in text:
                in_thinking = False
                before_close = text.split("</think>", 1)[0]
                after_close = text.split("</think>", 1)[1]
                if before_close:
                    yield {"reasoning": before_close}
                if after_close:
                    yield {"content": after_close}
                continue

            if in_thinking:
                yield {"reasoning": text}
            else:
                yield {"content": text}
