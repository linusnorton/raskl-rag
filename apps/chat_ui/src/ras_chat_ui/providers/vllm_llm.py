"""vLLM LLM provider — OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator

import httpx

from .base import LLMProvider

log = logging.getLogger(__name__)


class VLLMLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def count_tokens(self, messages: list[dict], *, tools: list[dict] | None = None) -> int:
        try:
            with httpx.Client(timeout=10.0) as client:
                base = self.base_url
                if base.endswith("/v1"):
                    base = base[:-3]
                payload: dict = {"model": self.model, "messages": messages}
                if tools:
                    payload["tools"] = tools
                resp = client.post(f"{base}/tokenize", json=payload)
                if resp.status_code == 200:
                    return resp.json()["count"]
        except Exception:
            pass
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
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            if resp.status_code != 200:
                log.error("LLM API error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        return {
            "tool_calls": msg.get("tool_calls"),
            "reasoning": msg.get("reasoning"),
            "content": msg.get("content"),
        }

    def chat_completion_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Generator[dict[str, str], None, None]:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        client = httpx.Client(timeout=300.0)
        resp = client.send(
            client.build_request("POST", f"{self.base_url}/chat/completions", json=payload),
            stream=True,
        )
        try:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                result: dict[str, str] = {}
                if delta.get("reasoning"):
                    result["reasoning"] = delta["reasoning"]
                if delta.get("content"):
                    result["content"] = delta["content"]
                if result:
                    yield result
        finally:
            resp.close()
