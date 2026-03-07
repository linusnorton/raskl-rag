# ras-chat-ui — RAG Chat Interface

## What this app does

`ras-chat-ui` is a Gradio web interface that lets users ask questions about the indexed documents.
It uses a technique called RAG (Retrieval-Augmented Generation): instead of relying on the LLM's
training data (which doesn't include these documents), the system retrieves relevant passages from
the indexed collection and provides them to the LLM as context. The LLM is instructed to answer
only from those passages.

## What is RAG?

Large language models are trained on general text from the internet. They don't know about
documents that weren't in their training data — and even if they did, they might misremember or
hallucinate details. RAG solves this by:

1. Converting the user's question into a vector (a list of numbers representing its meaning)
2. Finding the document passages whose vectors are most similar to the question vector
3. Giving those passages to the LLM and telling it: "answer using only these sources"

The LLM's role is to synthesise the retrieved passages into a coherent answer with citations, not
to recall information from memory.

## How it fits into the pipeline

```
PostgreSQL + pgvector   (populated by ras-chunker-indexer)
    │
    ▼
ras-chat-ui  ← this app
    │  user asks question
    │  → embed question → hybrid search → rerank → build prompt → stream response
    ▼
Gradio web UI  (http://localhost:7860)
```

## Architecture overview

Each user message goes through this process:

1. **Embed the query** — convert the question text to a 1024-dimension vector
2. **Hybrid search** — fetch up to 30 candidate passages via vector + full-text RRF search
3. **Rerank** — score all 30 candidates against the query; keep top 15
4. **Build prompt** — insert context passages as numbered `[1]...[15]` blocks into the system prompt
5. **Trim to context budget** — drop lowest-scoring chunks if the prompt would exceed the LLM's context window
6. **Tool-calling loop** — LLM may call `search_documents` or `web_search` up to 5 times for additional context
7. **Stream final answer** — stream the LLM response to the UI
8. **Renumber citations** — remap `[N]` indices to consecutive numbering in first-appearance order

## Key design decisions

### D1 — Grounding via system prompt

**What we chose:** The LLM receives a system prompt that emphasises narrative synthesis
(abbreviated):

> Write a narrative answer that synthesises information from the numbered context passages.
> Connect facts across multiple sources to build a coherent account. Where sources offer
> different perspectives or details, weave them together rather than listing them separately.
> Cite sources using [N] after the relevant sentence or clause. Every factual claim must have
> a citation, but integrate them naturally into prose — do not just list references.
> Draw on ALL relevant context passages. Preserve original spellings, names, and dates.
> All passage types are equally valid. Do not invent facts. Only state what the sources say.

Temperature is set to 0.5. The grounding prompt and thinking mode prevent hallucination while
allowing natural, connected prose. The earlier version (temperature 0.3, extraction-focused
prompt) produced terse bullet-point answers that listed citations rather than synthesising a
narrative.

**Why the change from extraction to synthesis:** The original prompt optimised for fact-finding
("Cite every factual claim using [N] references") which caused the LLM to act as a reference
lister rather than a historian. The new prompt asks the model to "connect facts across sources"
and "weave them together", producing longer, more coherent answers that are more useful for
research.

**Why:** The documents are historical sources — 19th- and 20th-century journals, correspondence,
and field notes. Hallucinated dates, names, or attributions would be directly harmful to research.

**Why no explicit refusal template:** An earlier version included the instruction "If the context
does not contain the answer, say: 'The provided documents do not contain information about this.'"
This caused Qwen3-235B to over-trigger the refusal — when the model had a convenient canned
refusal phrase, it would refuse even when the context clearly contained the answer (e.g. refusing
to discuss Abdullah's critique of early Malay accounts because the citation said "Abdullah" and
the query said "A. Rahman Tang Abdullah"). Removing the template forces the model to actually
evaluate the context before deciding it cannot answer.

**Why the surname matching rule:** Document metadata often lacks full author names — only
surnames are available (parsed from filenames like `Abdullah (2011) JMBRAS 84(1), 1-22.pdf`).
Without the explicit instruction to match by surname, the LLM treats "Abdullah" and "A. Rahman
Tang Abdullah" as different people and refuses to answer.

**Why the broad citation rule:** An earlier version of the prompt led the LLM to overlook
biographical notes and introductory sections as "meta" content rather than source material. The
explicit instruction that all passage types are equally valid sources fixed this.

### D2 — Agentic tool-calling loop

**What we chose:** The LLM can call tools up to 5 rounds before being forced to produce a final
answer. Two tools are available:

- **`search_documents`** — performs another hybrid search with a new query string. The results
  are added to the conversation context.
- **`web_search`** — searches DuckDuckGo (top 5 results) for general knowledge outside the
  indexed collection (enabled by default, can be disabled via `CHAT_WEB_SEARCH_ENABLED=false`).

**Streaming split:** Tool rounds run in non-streaming mode (the LLM must return tool calls as
complete JSON). Only the final answer — after all tool rounds — is streamed to the UI. This
means users see a brief pause during tool execution, then the answer streams in.

**Context budget management:** Three mechanisms keep the prompt within the LLM's context window:

1. **Pre-loop trimming:** Before the tool loop starts, the lowest-relevance chunks are dropped
   if the initial retrieval results would overflow the budget.
2. **Per-round check:** Before each tool round, the system checks whether accumulated context
   fits. If it doesn't, lowest-ranked chunks are dropped.
3. **Post-loop cleanup:** If context is nearly full after tool rounds, the oldest
   assistant + tool message pairs are deleted to make room for the final streaming response.

The context budget formula reserves space for both output and thinking tokens:
`max_tokens = min(llm_max_tokens + llm_thinking_budget, available - 64)`.

**Why an agent loop:** Initial retrieval might not find the best passages if the user's question
uses different phrasing from the document text. The LLM can recognise this from the retrieved
context and call `search_documents` again with a more targeted query. In practice, most queries
are answered in one retrieval round; the agent loop provides a fallback for complex multi-part
questions.

### D3 — Two-stage retrieval: hybrid search then reranker

**What we chose:**

- **Stage 1:** Fetch 30 candidates via hybrid RRF search (vector + full-text, see
  `apps/chunker_indexer/README.md` D4 for the full explanation of the search algorithm)
- **Stage 2:** Score all 30 candidates with a reranker model, return the top 15

Each candidate sent to the reranker is prefixed with document metadata:
```
Author: {author name}
Document: {title}
{chunk text}
```

**Why a reranker:** Vector search and RRF ranking are fast but imprecise. They measure how
similar a passage is to the query in embedding space, but embeddings compress meaning and lose
nuance. A reranker is a smaller language model trained specifically to compare a query and a
document and output a relevance score — it does a deeper analysis than embedding similarity.
Running a reranker on 30 candidates is fast (seconds), and it substantially improves precision:
the top 10 after reranking are noticeably more relevant than the top 10 from raw search alone.

**FTS AND-with-fallback:** The full-text search component uses a two-pass strategy. It first
tries AND-based matching (all query terms must appear). If AND returns >= 10 matches, those are
used. If fewer than 10, it falls back to OR-based matching (any term matches). This improves
precision for multi-term queries like "Birch telegram Penang" where AND finds exactly the right
passages, while preserving recall for queries where not all terms appear in the same chunk.

**Why prepend title and author to each chunk:** A chunk of text might be highly relevant in the
context of one document but less so from another. The reranker has no knowledge of which document
a chunk came from unless we tell it. Prepending the metadata gives the reranker context to make
better relevance judgements (e.g. a passage about Singapore from Swettenham's own journal should
score higher than the same words from an abstract).

### D4 — Citation renumbering

**What we chose:** The LLM receives context passages as `[1]...[10]`. After streaming its
response, the system:

1. Scans the response for all cited indices (e.g. `[1]`, `[5]`, `[8]`)
2. Remaps them to consecutive indices in first-appearance order (so `[5]` becomes `[1]`, `[8]`
   becomes `[2]`, etc.)
3. Applies the remapping using sentinel characters to prevent cascading replacement (e.g. `[5]`
   → `[\x005\x00]` → `[1]`)

The `<think>...</think>` block that Qwen3 models produce before their final answer is excluded from
citation processing. The thinking block's internal references (the model reasoning through which
passages are relevant) would otherwise pollute the Sources list with phantom citations.

**Why renumber:** If the LLM only cites passages 1, 3, and 8 of the 10 provided, the Sources
section would show `[1]`, `[3]`, `[8]` — a confusing, non-consecutive list for the reader. After
renumbering, the response shows `[1]`, `[2]`, `[3]` with Sources listed in that order. This is
standard citation practice.

### D5 — Thinking/reasoning blocks

**What we chose:** Qwen3 models produce a `<think>...</think>` block before their final answer.
The Gradio `Chatbot` component is configured with `reasoning_tags=[("<think>", "</think>")]`,
which renders these as a collapsible "Reasoning" section separate from the main message.

During streaming, reasoning tokens and content tokens arrive in separate fields from the vLLM
API. The UI reconstructs the full `<think>...</think>` prefix incrementally, so the user sees
thinking appearing in real time.

**Why show it:** Transparency. Users can expand the reasoning block to see how the model
interpreted their question and which sources it considered before answering. This helps users
judge whether the model followed the grounding rules correctly.

### D6 — Pluggable provider architecture

**What we chose:** Three independent provider interfaces, each with local and cloud implementations:

| Provider | Local backend | Cloud backend |
|----------|---------------|---------------|
| LLM | `vllm` — httpx to vLLM OpenAI-compatible API | `bedrock` — AWS Bedrock Converse API (Qwen3-235B) |
| Embedding | `sentence-transformers` — loads model locally | `bedrock` — Amazon Titan Embed Text v2 |
| Reranker | `qwen3` or `cross-encoder` — local models | `bedrock` — Amazon Rerank v1 |

Selected via `CHAT_LLM_PROVIDER`, `CHAT_EMBED_PROVIDER`, `CHAT_RERANK_PROVIDER`.

**Why:** Local GPU deployment keeps data on-premise and has no per-query API cost, but requires
adequate hardware. AWS Bedrock requires no local hardware but costs money per request and sends
data to AWS. The provider pattern means the same application code handles both cases; only
environment variables change between deployments.

### D7 — Extended thinking for Bedrock LLM

**What we chose:** When using Bedrock, the LLM can be given a thinking token budget via
`CHAT_LLM_THINKING_BUDGET` (set to 2048 in production). This passes
`additionalModelRequestFields.thinking` to the Bedrock Converse API, enabling the model to
reason through dates, facts, and cross-references before producing its answer.

Thinking tokens and output tokens share the same `max_tokens` budget — the context budget
formula adds them together (`llm_max_tokens + llm_thinking_budget`) and reserves that total
from the available context window. During streaming, reasoning tokens and content tokens arrive
in separate fields and are rendered independently (reasoning as a collapsible block, content
as the main answer).

**Why:** Without thinking mode, Qwen3 on Bedrock would not carefully parse dates and facts from
context passages. In testing, the model confused "6th October" with dates from adjacent section
headings, hallucinating "1st November" as an answer. With thinking enabled, the model reasons
through "the section heading says 3rd September to 6th October... he left for Singapore on the
5th... arrived 6th October" and gets the correct answer.

### D8 — Reranker domain hint for Bedrock

**What we chose:** The `CHAT_RERANK_INSTRUCTION` config value is prepended to the query before
sending it to the Bedrock reranker. For example, the query "What dates did Swettenham go to
Singapore?" becomes "Given a user question about historical JMBRAS and Swettenham journal
documents, judge whether the document passage is relevant: What dates did Swettenham go to
Singapore?"

This reuses the same `rerank_instruction` field used by the local Qwen3 reranker (where it's
passed as a model instruction). For Bedrock rerankers that don't support custom instructions
(like Cohere Rerank and Amazon Rerank), prepending to the query is the workaround.

**Why:** Without domain context, the reranker treats all queries generically. Adding a hint about
the document domain helps the reranker prioritise passages from the right context — a passage
about Singapore from Swettenham's journal should score higher than a generic mention of Singapore
in an unrelated paper.

## Configuration

Key environment variables (prefix `CHAT_`):

### Provider selection

| Variable | Default | Options |
|----------|---------|---------|
| `CHAT_LLM_PROVIDER` | `vllm` | `vllm`, `bedrock` |
| `CHAT_EMBED_PROVIDER` | `sentence-transformers` | `sentence-transformers`, `bedrock` |
| `CHAT_RERANK_PROVIDER` | `qwen3` | `qwen3`, `cross-encoder`, `bedrock` |

### LLM

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_LLM_BASE_URL` | `http://localhost:8002/v1` | vLLM OpenAI-compatible endpoint |
| `CHAT_LLM_MODEL` | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | Heavy stack default |
| `CHAT_LLM_MAX_TOKENS` | `4096` | Maximum tokens to generate |
| `CHAT_LLM_CONTEXT_WINDOW` | `40960` | Total context budget (prompt + generation) |
| `CHAT_LLM_TEMPERATURE` | `0.5` | Balanced for narrative synthesis with grounding |

### Embedding

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_EMBED_MODEL` | `./models/Qwen--Qwen3-Embedding-8B` | Local path format |
| `CHAT_EMBED_DIMENSIONS` | `1024` | Must match indexed embedding dimensions |
| `CHAT_EMBED_TASK_PREFIX` | `"Represent this query for retrieving relevant passages: "` | Empty `""` for BGE-M3 |
| `CHAT_EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |

### Reranking

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_RERANK_MODEL` | `./models/Qwen--Qwen3-Reranker-8B` | Local path format |
| `CHAT_RERANK_ENABLED` | `true` | Disable to skip reranking (not recommended) |
| `CHAT_RERANK_CANDIDATES` | `30` | Candidates fetched before reranking |
| `CHAT_RERANK_BACKEND` | `qwen3` | `qwen3` (causal LM) or `cross-encoder` |
| `CHAT_RERANK_DEVICE` | `cpu` | `cpu` or `cuda` |

### Retrieval

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_RETRIEVAL_TOP_K` | `15` | Final passages returned to LLM after reranking |

### AWS Bedrock (cloud deployment)

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_BEDROCK_REGION` | `eu-west-2` | AWS region |
| `CHAT_BEDROCK_CHAT_MODEL_ID` | `qwen.qwen3-235b-a22b-2507-v1:0` | Bedrock LLM model ID |
| `CHAT_BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `CHAT_BEDROCK_RERANK_REGION` | `eu-central-1` | AWS region for reranking (may differ) |
| `CHAT_BEDROCK_RERANK_MODEL_ID` | `amazon.rerank-v1:0` | Bedrock reranker model |
| `CHAT_LLM_THINKING_BUDGET` | `2048` | Extended thinking token budget (0=disabled) |
| `CHAT_RERANK_INSTRUCTION` | _(historical JMBRAS domain hint)_ | Prepended to rerank queries for domain context |

### Database

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_DB_NAME` | `raskl_rag` | PostgreSQL database name |
| `CHAT_DATABASE_DSN` | _(empty)_ | Override full connection string (e.g. Neon serverless) |

### Other

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_WEB_SEARCH_ENABLED` | `true` | Enables the DuckDuckGo `web_search` tool |
| `CHAT_GRADIO_PORT` | `7860` | Gradio server port |

See root `CLAUDE.md` for all CLI commands and startup scripts.

## Testing

The chat UI is tested manually via the Gradio interface. The retrieval and reranking components
are tested indirectly via the chunker_indexer tests and end-to-end pipeline runs.

Key checks after any change to retrieval or prompting:

1. Query a known entity (e.g. a person mentioned in a specific document) — check the correct
   passage is retrieved
2. Query something not in the collection — check the UI responds "The provided documents do not
   contain information about this" rather than hallucinating
3. Check that citations in the response match the Sources list and are numbered consecutively
4. Check that page numbers in the Sources list match the journal's actual page numbers (not PDF
   viewer page numbers)
