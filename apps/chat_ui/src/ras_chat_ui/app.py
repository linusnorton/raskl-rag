"""Gradio chat interface for RAG queries over JMBRAS documents."""

from __future__ import annotations

import os
import secrets
from collections.abc import Generator

import fastapi
import gradio as gr
import gradio.networking

from ras_rag_engine.agent import run_agent_streaming
from ras_rag_engine.citations import renumber_response
from ras_rag_engine.config import RAGConfig
from ras_rag_engine.retriever import RetrievedChunk


class ChatUIConfig(RAGConfig):
    """Extends RAG config with Gradio-specific settings."""

    gradio_port: int = 7860
    gradio_share: bool = False
    gradio_password: str = "Swettenham"


def _make_chat_fn(config: ChatUIConfig):
    """Create the chat function closed over config."""

    def chat_fn(message: str, history: list[dict]) -> Generator[str, None, None]:
        """Stream responses from the RAG agent."""
        all_chunks: list[RetrievedChunk] = []
        partial_text = ""

        try:
            for partial_text, chunks in run_agent_streaming(message, history, config):
                all_chunks = chunks
                yield partial_text
        except Exception as e:
            yield f"Error: {e}"
            return

        final = renumber_response(partial_text, all_chunks)
        if final != partial_text:
            yield final

    return chat_fn


def main() -> None:
    """CLI entry point: launch the Gradio chat UI."""
    config = ChatUIConfig()

    chatbot = gr.Chatbot(reasoning_tags=[("<think>", "</think>")])
    demo = gr.ChatInterface(
        fn=_make_chat_fn(config),
        chatbot=chatbot,
        title="SwetBot",
        description="Ask questions about JMBRAS and Swettenham historical documents.",
    )
    # Use HTTP Basic Auth via auth_dependency — stateless, so it works across
    # Lambda cold starts (Gradio's built-in auth uses in-memory tokens which are
    # lost when a new Lambda instance starts).
    auth_dep = None
    if config.gradio_password:
        expected_user = "swetbot"
        expected_pass = config.gradio_password

        def _check_basic_auth(request: fastapi.Request) -> str | None:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                import base64
                try:
                    decoded = base64.b64decode(auth[6:]).decode()
                    user, password = decoded.split(":", 1)
                except Exception:
                    pass
                else:
                    if secrets.compare_digest(user, expected_user) and secrets.compare_digest(password, expected_pass):
                        return user
            # Trigger browser's native login dialog
            raise fastapi.HTTPException(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="SwetBot"'},
            )

        auth_dep = _check_basic_auth

    # In Lambda, Gradio's self-connectivity check fails because the server isn't
    # reachable via localhost during cold start. Skip the check — the Lambda Web
    # Adapter handles readiness via AWS_LWA_READINESS_CHECK_PATH instead.
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        gradio.networking.url_ok = lambda url: True
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.gradio_port,
        share=config.gradio_share,
        auth_dependency=auth_dep,
    )


if __name__ == "__main__":
    main()
