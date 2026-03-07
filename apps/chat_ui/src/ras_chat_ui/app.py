"""Gradio chat interface for RAG queries over JMBRAS documents."""

from __future__ import annotations

import re
from collections.abc import Generator

import os

import gradio as gr
import gradio.networking

from .agent import run_agent_streaming
from .config import ChatConfig
from .retriever import RetrievedChunk


def _extract_cited_indices(text: str) -> set[int]:
    """Extract [N] citation indices from LLM response text."""
    return {int(m) for m in re.findall(r'\[(\d+)\]', text)}


def _extract_content(text: str) -> str:
    """Strip <think>...</think> block and return only the final response body."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


def _make_index_map(text: str) -> dict[int, int]:
    """Map original [N] citation indices to consecutive display numbers (in order of first appearance)."""
    seen: dict[int, int] = {}
    for m in re.finditer(r'\[(\d+)\]', text):
        orig = int(m.group(1))
        if orig not in seen:
            seen[orig] = len(seen) + 1
    return seen


def _renumber_text(text: str, mapping: dict[int, int]) -> str:
    """Replace [N] citations in text using consecutive display numbers via sentinel to avoid cascading replacements."""
    for orig, display in mapping.items():
        text = text.replace(f"[{orig}]", f"[\x00{display}\x00]")
    return text.replace("[\x00", "[").replace("\x00]", "]")


def _format_citations(chunks: list[RetrievedChunk], response_text: str = "", index_map: dict[int, int] | None = None) -> str:
    """Build a deduplicated markdown citation block from retrieved chunks.

    Only includes chunks that the LLM actually cited via [N] markers.
    If no cited indices are found, falls back to showing all chunks.
    index_map remaps original chunk indices to consecutive display labels.
    """
    if not chunks:
        return ""

    cited = _extract_cited_indices(response_text) if response_text else set()

    seen: set[str] = set()
    lines: list[str] = []
    display_n = 0
    for i, c in enumerate(chunks, start=1):
        # Filter to only cited chunks (if we detected any citations)
        if cited and i not in cited:
            continue

        key = (c.chunk_id, c.start_page)
        if key in seen:
            continue
        seen.add(key)

        display_n += 1
        label = index_map[i] if index_map and i in index_map else display_n

        source = c.title or c.source_filename
        display_start = c.start_page
        display_end = c.end_page
        pages = f"p.{display_start}" if display_start == display_end else f"pp.{display_start}-{display_end}"
        heading = f" — {c.section_heading}" if c.section_heading else ""
        author_year = ""
        if c.author or c.year:
            parts = []
            if c.author:
                parts.append(c.author)
            if c.year:
                parts.append(str(c.year))
            author_year = f" ({', '.join(parts)})"
        lines.append(f"[{label}] {source}{author_year}, {pages}{heading}")

    if not lines:
        return ""
    # Sort by display label so sources appear in [1], [2], [3] order
    lines.sort(key=lambda l: int(re.match(r'\[(\d+)\]', l).group(1)))
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def _make_chat_fn(config: ChatConfig):
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

        # Renumber citations in the body (not the thinking block) and build sources from original indices
        content = _extract_content(partial_text)
        mapping = _make_index_map(content)
        if mapping:
            renumbered = _renumber_text(content, mapping)
            think_match = re.match(r'<think>.*?</think>\s*', partial_text, re.DOTALL)
            partial_text = (think_match.group(0) if think_match else '') + renumbered
        # Pass original content (pre-renumber) so cited filter uses original LLM chunk indices
        citations = _format_citations(all_chunks, content, index_map=mapping)
        if citations and partial_text:
            yield partial_text + citations

    return chat_fn


def main() -> None:
    """CLI entry point: launch the Gradio chat UI."""
    config = ChatConfig()

    chatbot = gr.Chatbot(reasoning_tags=[("<think>", "</think>")])
    demo = gr.ChatInterface(
        fn=_make_chat_fn(config),
        chatbot=chatbot,
        title="raskl-rag Chat",
        description="Ask questions about JMBRAS and Swettenham historical documents.",
    )
    auth = ("raskl", config.gradio_password) if config.gradio_password else None
    # In Lambda, Gradio's self-connectivity check fails because the server isn't
    # reachable via localhost during cold start. Skip the check — the Lambda Web
    # Adapter handles readiness via AWS_LWA_READINESS_CHECK_PATH instead.
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        gradio.networking.url_ok = lambda url: True
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.gradio_port,
        share=config.gradio_share,
        auth=auth,
    )


if __name__ == "__main__":
    main()
