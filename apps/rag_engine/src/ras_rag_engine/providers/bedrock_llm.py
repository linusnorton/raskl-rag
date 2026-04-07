"""AWS Bedrock LLM provider via Converse API."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator

from .base import LLMProvider

log = logging.getLogger(__name__)

_client = None


def _get_client(region: str):
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-runtime", region_name=region)
    return _client


def _openai_tools_to_bedrock(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Bedrock toolSpec format."""
    bedrock_tools = []
    for t in tools:
        fn = t["function"]
        bedrock_tools.append({
            "toolSpec": {
                "name": fn["name"],
                "description": fn["description"],
                "inputSchema": {"json": fn["parameters"]},
            }
        })
    return bedrock_tools


def _convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split messages into Bedrock system + messages format.

    Bedrock Converse API requires:
    - system messages go into the `system` parameter
    - tool results use `toolResult` content blocks
    - assistant tool_calls become `toolUse` content blocks
    """
    system_parts: list[dict] = []
    bedrock_messages: list[dict] = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append({"text": msg["content"]})

        elif role == "user":
            bedrock_messages.append({"role": "user", "content": [{"text": msg["content"]}]})

        elif role == "assistant":
            content_blocks = []
            if msg.get("content") and str(msg["content"]).strip():
                content_blocks.append({"text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                content_blocks.append({
                    "toolUse": {
                        "toolUseId": tc["id"],
                        "name": fn["name"],
                        "input": args,
                    }
                })
            if content_blocks:
                bedrock_messages.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            tool_result = {
                "toolResult": {
                    "toolUseId": msg["tool_call_id"],
                    "content": [{"text": msg["content"]}],
                }
            }
            
            if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                bedrock_messages[-1]["content"].append(tool_result)
            else:
                bedrock_messages.append({"role": "user", "content": [tool_result]})

    return system_parts, bedrock_messages


class BedrockLLMProvider(LLMProvider):
    def __init__(self, region: str, model_id: str, thinking_budget: int = 0):
        self.region = region
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
        client = _get_client(self.region)
        system_parts, bedrock_messages = _convert_messages(messages)

        kwargs: dict = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            kwargs["system"] = system_parts
        if tools:
            kwargs["toolConfig"] = {"tools": _openai_tools_to_bedrock(tools)}
        resp = client.converse(**kwargs)
        output = resp["output"]["message"]

        # Parse response into normalized format
        tool_calls = None
        content = None
        reasoning = None

        for block in output.get("content", []):
            if "text" in block:
                content = block["text"]
            elif "toolUse" in block:
                if tool_calls is None:
                    tool_calls = []
                tu = block["toolUse"]
                tool_calls.append({
                    "id": tu["toolUseId"],
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu["input"]),
                    },
                })
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"].get("reasoningText", {}).get("text")

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
        client = _get_client(self.region)
        system_parts, bedrock_messages = _convert_messages(messages)

        kwargs: dict = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            kwargs["system"] = system_parts
        if tools:
            kwargs["toolConfig"] = {"tools": _openai_tools_to_bedrock(tools)}
        resp = client.converse_stream(**kwargs)

        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                result: dict[str, str] = {}
                if "text" in delta:
                    result["content"] = delta["text"]
                elif "reasoningContent" in delta:
                    text = delta["reasoningContent"].get("text", "")
                    if text:
                        result["reasoning"] = text
                if result:
                    yield result
