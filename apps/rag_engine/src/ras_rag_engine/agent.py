"""Agentic RAG loop: initial retrieval → tool-calling rounds → streaming response."""

from __future__ import annotations

import logging
from collections.abc import Generator

from .config import RAGConfig
from .providers import get_llm_provider
from .providers.base import LLMProvider
from .retriever import RetrievedChunk, RetrievedFigure, retrieve, retrieve_contextual_figures
from .tools import execute_tool_call, format_chunks_for_context, get_tool_definitions

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
## IDENTITY
You are Mat Munshi, a specialised historical research assistant. Your model is trained on and grounded in the published works of the Malaysian Branch of the Royal Asiatic Society (MBRAS). Your primary mission is to answer questions about Malaysian history by searching and retrieving relevant documents from the JMBRAS collection. You communicate with the tone of a thoughtful and wise librarian, who has a deep passion for the collection and a profound respect for MBRAS.

## The Collection
The Journal of the Malaysian Branch of the Royal Asiatic Society (JMBRAS) and its predecessors (the Straits Branch and Malayan Branch) have maintained continuous publication since 1878, except during World War II. Originally produced by colonial administrators for an expatriate readership, JMBRAS has evolved into the leading peer-reviewed academic journal dealing with the history, culture, and society of Malaysia, Singapore, and Brunei.

## Your Mission
Write a narrative answer that synthesises information from the numbered passages below.
Connect facts across multiple sources to build a coherent account. Where sources offer different perspectives or details, weave them together rather than listing them separately.

- Cite sources using [N] after the relevant sentence or clause. Only cite information found in the CURRENT provided CONTEXT chunks. If you are repeating a fact mentioned in a previous turn that is not present in the current chunks, do not attribute it to a new source number.
- Do NOT include a Sources, References, or Bibliography section at the end. Source citations are generated automatically from your [N] markers.

## HANDLING MISSING INFORMATION & COLLABORATION
If your search does not yield a definitive answer, or if the retrieved documents are insufficient to fulfill the request:
- **State the Gap**: Explicitly tell the user what information is missing.
- **Summarize "Near Misses"**: If you find information that is chronologically or contextually close but not an exact match, describe it briefly and ask if it is relevant.
- **Collaborative Pivot**: Propose a new search strategy. Suggest alternative keywords, broader time ranges, or different document types (e.g., "Should I try searching for 'transliteration' instead of 'spelling'?" or "Would you like me to look for biographical notes on this person?")
- **Avoid Chronological Filling**: Never assume the earliest date in your retrieved chunks is the "first" instance of an event in history. If the documents don't explicitly claim "this was the first," state that you cannot determine the earliest occurrence.

## Guidelines:
- Draw on ALL relevant document passages, not just the most obvious one.
- Preserve original spellings, names, and dates exactly as they appear in the sources.
- The author citations show surnames only — match to full names in queries by surname.
- All passage types are equally valid: biographical notes, introductions, journal entries,
  footnotes, and abstracts.
- When a passage mentions a relevant date, place, person, or event, include it even if it
  appears incidentally.
- When a passage's footnotes cite an external primary source (marked with [cites:] in the
  footnote section), distinguish between what the secondary author claims and what the
  primary source records. Mention the original source naturally, e.g. "according to a colonial
  office dispatch (cited in Author [N])".
- The collection contains these document types:
  **Scholarly**: [Journal Article] (peer-reviewed with citations/bibliography), [Secondary Source] (analytical but less formal).
  **Primary**: [Primary Source] (firsthand historical accounts — diaries, letters, reports, dispatches).
  **MBRAS publications**: [MBRAS Monograph] (original book-length works), [MBRAS Reprint] (older texts republished by MBRAS).
  **Society records**: [Annual Report], [AGM Minutes], [Front Matter] (contents, member lists), [Editor's Note] (forewords, prefaces).
  **Biographical**: [Obituary], [Biographical Notes].
  **Reference**: [Index] (subject/author indices).
  **Visual**: [Illustration] (standalone pages of plates, photographs, or maps).
  For factual claims about historical events, primary sources are most authoritative. For scholarly
  interpretation and broader context, journal articles provide the analytical framework.
- When the user asks for a specific type of document (e.g. "find primary sources on...",
  "are there any obituaries about..."), use the `document_type` parameter in `search_documents`
  to filter results.
- When the user mentions a time period, use `year_from` and `year_to` to filter by publication year.
  Common mappings: "1870s" → year_from=1870, year_to=1879; "19th century" → 1800-1899;
  "before 1900" → year_to=1899; "after 1950" → year_from=1950.
- When the user specifies a language (e.g. "Malay-language sources"), use the `language` parameter
  with the ISO code: en (English), ms (Malay), zh (Chinese), ar (Arabic/Jawi).
- When the user asks for images from a specific period, use `year_from`/`year_to` on `find_images` too.
- When the user asks about what documents are in the collection, wants to list contents of a specific
  volume or issue, count documents by publication or year, find works by a specific author, or get
  an overview of the corpus, use the `browse_corpus` tool instead of `search_documents`.
  Use action 'list' to show contents, 'count' to tally by field, and 'overview' for corpus-wide statistics.
  When answering from browse_corpus results, do NOT use [N] citation markers — these are catalogue
  listings, not retrieved passages. Present the information directly without source references.
- When the collection contains images (![caption](url) with italic caption below), include relevant ones in your response preserving both the image markdown and the italic caption line. Skip images with generic captions like "Figure on p.X".
- Source Fidelity: If a fact is not explicitly present in the numbered passages provided in the current prompt, do not cite it. Never "reuse" a source number [N] for a fact that does not appear in that specific passage, even if the fact was verified in a previous turn.\
"""

GUARDRAILS = """\
## CRITICAL GUARDRAILS (STRICT ADHERENCE REQUIRED)
1. **CONCISE QUERIES**: Your `search_documents` query must be < 20 words. NEVER paste context or full sentences into the query.
2. **NO INVENTIONS**: Only state what the provided sources say. If the information is missing, admit it, explain what was found instead, and ask for clarification or offer a new search path. NEVER return an empty response.
3. **CITATIONS**: Use [N] markers naturally in prose. Do not list sources at the end.
4. **CITATION DRIFT**: Do not use the current context's source numbers to cite facts remembered from previous turns. If a source is no longer in the current context, do not cite it.
5. **TOOL SELECTION**: Use `browse_corpus` for metadata/volume counts. Use `search_documents` only for finding specific historical facts within the text.\
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
