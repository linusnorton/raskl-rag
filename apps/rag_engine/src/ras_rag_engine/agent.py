"""Agentic RAG loop: initial retrieval → tool-calling rounds → streaming response."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator

from .config import RAGConfig
from .providers import get_llm_provider
from .providers.base import LLMProvider
from .retriever import RetrievedChunk, RetrievedFigure, retrieve, retrieve_contextual_figures
from .tools import execute_tool_call, format_chunks_for_context, get_tool_definitions

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
## IDENTITY
You are Mat Munshi, a specialised historical research assistant. Your model is trained on and grounded in the published works of the Malaysian Branch of the Royal Asiatic Society (MBRAS). You communicate as a thoughtful, wise librarian with a deep passion for MBRAS.

## The Collection
The Journal of the Malaysian Branch of the Royal Asiatic Society (JMBRAS) and its predecessors (the Straits Branch and Malayan Branch) have maintained continuous publication since 1878, except during World War II. Originally produced by colonial administrators for an expatriate readership, JMBRAS has evolved into the leading peer-reviewed academic journal dealing with the history, culture, and society of Malaysia, Singapore, and Brunei.

- The collection consists of documents published by MBRAS and external documents. 
- Document types includes:
  **Scholarly**: [Journal Article] (peer-reviewed with citations/bibliography), [Secondary Source] (analytical but less formal).
  **Primary**: [Primary Source] (firsthand historical accounts — diaries, letters, reports, dispatches).
  **MBRAS publications**: [MBRAS Monograph] (original book-length works), [MBRAS Reprint] (older texts republished by MBRAS).
  **Society records**: [Annual Report], [AGM Minutes], [Front Matter] (contents, member lists), [Editor's Note] (forewords, prefaces).
  **Biographical**: [Obituary], [Biographical Notes].
  **Reference**: [Index] (subject/author indices).
  **Visual**: [Illustration] (standalone pages of plates, photographs, or maps).
  
## Your Mission

Follow this logic for every interaction:

1. **ASSESS & VERIFY**: Review initial context chunks.
  - **Grounding Gate**: If a user makes a premise, but your sources don't mention it, do not assume the premise is correct without verifying it.
  - **One-Shot Check**: If current documents provide a full, accurate answer, skip searching and respond immediately.
2. **GAP ANALYSIS & TARGETED SEARCH**: If information is missing:
  - State exactly what is missing in your thinking block.
  - Use the **Librarian's Toolbox** protocol below.
3. **SYNTHESIS**: Write a narrative answer synthesising the numbered passages [N].
  - Connect facts across sources. Cite using [N] to ground your assertions in the sources.
  - Do NOT include a Sources, References, or Bibliography section at the end.

## LIBRARIAN'S TOOLBOX
Follow this hierarchy to find information efficiently:
1. Discovery: mbras_index — Use first to find canonical author names or articles on a specific topic.
2. Recall: search_documents — Use for broad historical facts. Use 3-5 nouns/entities only.
3. Refinement: search_by_attribute — Use if results are too noisy to filter by year, publication, or document type.
4. Deep Reading: document_context — Use to read pages preceding/following a promising result. Do NOT perform a new vector search for context.
5. Precision/Meta: exact_keyword_search for rare spellings; browse_corpus for metadata/volume listings; find_images for visual records.

## HANDLING MISSING INFORMATION
- **State the Gap**: Explicitly tell the user what the collection is missing.
- **Summarize "Near Misses"**: Describe contextually close information and ask if it is relevant.
- **Collaborative Pivot**: Propose a new search strategy or ask the user to elaborate.

## Guidelines:
- Preserve original spellings, names, and dates exactly as they appear.
- Include images if present () with their italic captions.
- Maintain a helpful, scholarly, and non-robotic persona.
"""

GUARDRAILS = """\
## CRITICAL GUARDRAILS (STRICT ADHERENCE REQUIRED)
1. **CONCISE QUERIES**: All search queries must be **3-5 words**. Use nouns/entities only. NEVER search for questions or full sentences.
2. **ENTITY-FIRST SEARCH**: If searching for a person, the query should just be their name. Do not add descriptive words (e.g., search for "Isabella Bird", NOT "Isabella Bird meeting in Penang").
3. **NO INVENTIONS**: Only state what the provided sources say. Never "reuse" a source number [N] from the conversation history.
4. **DEEP READING**: If a chunk mentions a fact but is cut off, you MUST use `document_context` before giving up or performing a new vector search.
"""

MAX_TOOL_ROUNDS = 4
# Token budget reserved for the completion output
MIN_OUTPUT_TOKENS = 256


def _build_system_prompt(chunks: list[RetrievedChunk], figures: list[RetrievedFigure] | None = None) -> str:
    """Build system prompt with initial context chunks and trailing guardrails."""
    context = format_chunks_for_context(chunks, figures=figures)
    
    # Sandwich the context between the base identity and the strict guardrails
    return f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{context}\n\n---\n{GUARDRAILS}"


def _compute_max_tokens(
    messages: list[dict],
    llm: LLMProvider,
    config: RAGConfig,
    *,
    tools: list[dict] | None = None,
) -> int:
    """Compute dynamic max_tokens based on remaining context budget."""
    input_tokens = llm.count_tokens(messages, tools=tools)
    available = config.llm_context_window - input_tokens
    # Reserve space for thinking tokens (they count against max_tokens on Bedrock)
    max_tokens = min(config.llm_max_tokens + config.llm_thinking_budget, available - 64)
    return max(max_tokens, min(MIN_OUTPUT_TOKENS, available))


def run_agent_streaming(
    user_message: str,
    history: list[dict],
    config: RAGConfig,
) -> Generator[tuple[str, list[RetrievedChunk]], None, None]:
    """Run the agentic RAG loop, yielding (partial_text, all_chunks) tuples.

    The generator yields incremental text tokens during streaming, and the final
    yield includes all retrieved chunks for citation display.
    """
    llm = get_llm_provider(config)
    tool_defs = get_tool_definitions(config)

    # Step 1: Initial RAG retrieval
    initial_chunks = retrieve(user_message, config)
    all_chunks: list[RetrievedChunk] = list(initial_chunks)

    # Step 1b: Retrieve contextual figures for the initial chunks
    try:
        contextual_figures = retrieve_contextual_figures(initial_chunks, config)
    except Exception:
        log.warning("Failed to retrieve contextual figures", exc_info=True)
        contextual_figures = []

    # Step 2: Build messages
    system_msg = {"role": "system", "content": _build_system_prompt(initial_chunks, contextual_figures)}
    messages: list[dict] = [system_msg]
    recent_history = history[-6:]
    for entry in recent_history:
        content = entry.get("content") or ""
        # Handle content as list of text blocks (OpenAI format)
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        if entry["role"] == "assistant":
            content = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', content)
        messages.append({"role": entry["role"], "content": content})
    messages.append({"role": "user", "content": user_message})

    # Step 2b: Trim context — drop lowest-relevance chunks until prompt fits
    budget = config.llm_context_window - MIN_OUTPUT_TOKENS - config.llm_thinking_budget - 64
    input_tokens = llm.count_tokens(messages, tools=tool_defs)
    while input_tokens > budget and initial_chunks:
        dropped = initial_chunks.pop()  # last chunk = lowest relevance score
        log.info("Dropping chunk (score=%.3f) to fit context: %d > %d", dropped.score, input_tokens, budget)
        messages[0] = {"role": "system", "content": _build_system_prompt(initial_chunks, contextual_figures)}
        input_tokens = llm.count_tokens(messages, tools=tool_defs)

    # Step 3: Tool-calling loop (non-streaming)
    for _round in range(MAX_TOOL_ROUNDS):
        # Check if context is too full for another tool round — skip to final answer
        input_tokens = llm.count_tokens(messages, tools=tool_defs)
        if input_tokens + MIN_OUTPUT_TOKENS + 64 > config.llm_context_window:
            log.info("Context nearly full (%d tokens), skipping to final answer", input_tokens)
            break

        max_tokens = _compute_max_tokens(messages, llm, config, tools=tool_defs)
        result = llm.chat_completion(
            messages, max_tokens=max_tokens, temperature=config.llm_temperature, tools=tool_defs
        )

        tool_calls = result.get("tool_calls")
        if not tool_calls:
            # No tool calls — this is the final text response (non-streamed path)
            reasoning = result.get("reasoning") or ""
            content = result.get("content") or ""
            text = f"<think>{reasoning}</think>{content}" if reasoning else content
            yield text, all_chunks
            return

        # Append the assistant message with tool calls (OpenAI format for message history)
        assistant_msg: dict = {"role": "assistant", "content": result.get("content") or ""}
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Execute each tool call
        tool_results_blocks = []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            result_text, new_chunks = execute_tool_call(fn_name, fn_args, config, start_index=len(all_chunks) + 1)
            # Keep ALL chunks (including duplicates) so that positions match the
            # [N] numbers the LLM sees in tool results.  format_citations()
            # deduplicates by chunk_id when building the Sources list.
            all_chunks.extend(new_chunks)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })

    # Step 4: Stream the final response (after tool rounds, or if we exhausted rounds)
    # Drop oldest tool-round messages if context is too full for a response
    stream_tools = None
    while len(messages) > 2:
        input_tokens = llm.count_tokens(messages)
        if input_tokens + MIN_OUTPUT_TOKENS <= config.llm_context_window:
            break
        # Find and remove the oldest assistant+tool group (after system msg)
        removed = False
        for i in range(1, len(messages)):
            if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                # Remove this assistant message and all following tool messages
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                del messages[i:j]
                removed = True
                break
        if not removed:
            break

    # Pass tools if message history contains tool_calls (Bedrock requires toolConfig)
    has_tool_messages = any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)
    stream_tools = tool_defs if has_tool_messages else None
    max_tokens = _compute_max_tokens(messages, llm, config, tools=stream_tools)
    accumulated_reasoning = ""
    accumulated_content = ""

    for delta in llm.chat_completion_stream(
        messages, max_tokens=max_tokens, temperature=config.llm_temperature, tools=stream_tools
    ):
        reasoning_token = delta.get("reasoning", "")
        content_token = delta.get("content", "")
        if reasoning_token:
            accumulated_reasoning += reasoning_token
            yield f"<think>{accumulated_reasoning}", all_chunks
        if content_token:
            accumulated_content += content_token
            if accumulated_reasoning:
                yield f"<think>{accumulated_reasoning}</think>{accumulated_content}", all_chunks
            else:
                yield accumulated_content, all_chunks
