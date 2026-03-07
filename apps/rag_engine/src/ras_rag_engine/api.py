"""OpenAI-compatible chat completions API backed by the RAG engine."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import run_agent_streaming
from .citations import renumber_response
from .config import RAGConfig
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

app = FastAPI(title="SwetBot RAG API")
_config: RAGConfig | None = None
_security = HTTPBearer(auto_error=False)


def _get_config() -> RAGConfig:
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def _check_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    config: RAGConfig = Depends(_get_config),
) -> None:
    if not config.api_key:
        return
    if credentials is None or credentials.credentials != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "swetbot"
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None


@app.get("/v1/models")
def list_models(config: RAGConfig = Depends(_get_config)):
    return {
        "object": "list",
        "data": [
            {
                "id": "swetbot",
                "object": "model",
                "created": 0,
                "owned_by": "raskl-rag",
            }
        ],
    }


def _build_chunk_event(
    completion_id: str,
    content: str = "",
    finish_reason: str | None = None,
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return json.dumps(chunk)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: None = Depends(_check_auth),
    config: RAGConfig = Depends(_get_config),
):
    messages = [m.model_dump() for m in request.messages]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    user_message = messages[-1]["content"]
    history = messages[:-1]

    if request.temperature is not None:
        config = config.model_copy(update={"llm_temperature": request.temperature})

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not request.stream:
        return _non_streaming_response(user_message, history, config, completion_id)

    return EventSourceResponse(
        _stream_response(user_message, history, config, completion_id),
        media_type="text/event-stream",
    )


def _non_streaming_response(
    user_message: str,
    history: list[dict],
    config: RAGConfig,
    completion_id: str,
) -> dict:
    all_chunks: list[RetrievedChunk] = []
    final_text = ""
    for partial_text, chunks in run_agent_streaming(user_message, history, config):
        final_text = partial_text
        all_chunks = chunks

    final_text = renumber_response(final_text, all_chunks)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_response(
    user_message: str,
    history: list[dict],
    config: RAGConfig,
    completion_id: str,
):
    """Yield SSE events in OpenAI streaming format."""
    # Initial role event
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield {"data": json.dumps(role_chunk)}

    all_chunks: list[RetrievedChunk] = []
    prev_text = ""

    try:
        for partial_text, chunks in run_agent_streaming(user_message, history, config):
            all_chunks = chunks
            # Send incremental content delta
            if len(partial_text) > len(prev_text):
                delta = partial_text[len(prev_text):]
                prev_text = partial_text
                yield {"data": _build_chunk_event(completion_id, content=delta)}

        # After streaming completes, send citation block as final content
        final_with_citations = renumber_response(prev_text, all_chunks)
        if len(final_with_citations) > len(prev_text):
            citation_delta = final_with_citations[len(prev_text):]
            yield {"data": _build_chunk_event(completion_id, content=citation_delta)}

    except Exception as e:
        log.exception("Error during streaming")
        yield {"data": _build_chunk_event(completion_id, content=f"\n\nError: {e}")}

    yield {"data": _build_chunk_event(completion_id, finish_reason="stop")}
    yield {"data": "[DONE]"}


def main() -> None:
    """CLI entry point: launch the API server."""
    import uvicorn

    config = _get_config()
    log.info("Starting RAG API on port %d", config.api_port)
    uvicorn.run(app, host="0.0.0.0", port=config.api_port)


if __name__ == "__main__":
    main()
