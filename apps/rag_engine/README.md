# ras-rag-engine — RAG API

## What this app does

`ras-rag-engine` is an OpenAI-compatible chat completions API backed by a retrieval-augmented
generation pipeline. It retrieves relevant passages from a PostgreSQL/pgvector index, feeds them
as context to a large language model, and returns a cited narrative response.

The API is designed to be consumed by Open WebUI (or any OpenAI-compatible client) and includes
endpoints for chat, speech-to-text (AWS Transcribe), and text-to-speech (Amazon Polly).

## How it fits into the pipeline

```
PostgreSQL + pgvector  (indexed by ras-chunker-indexer)
    │
    ▼
ras-rag-engine  ← this app
    │  /v1/chat/completions (OpenAI-compatible)
    │  /v1/audio/transcriptions (STT)
    │  /v1/audio/speech (TTS)
    ▼
Open WebUI  (chat frontend)
```

## Architecture overview

A chat request flows through these stages:

1. **Retrieve** — embed the user query, run hybrid search (vector + full-text RRF), optionally
   rerank
2. **Build prompt** — system prompt + retrieved context passages + conversation history
3. **Tool-calling loop** — LLM may call `search_documents` or `web_search` for more context
   (up to 5 rounds)
4. **Stream response** — stream the final LLM response with incremental tokens
5. **Citations** — extract `[N]` markers, renumber consecutively, append formatted source list

## Key design decisions

### D1 — Hybrid search: vector + full-text with Reciprocal Rank Fusion

**What we chose:** Two independent searches run in parallel on each query:

- **Vector search:** cosine similarity between the query embedding and all stored chunk
  embeddings, top 50 candidates
- **Full-text search:** PostgreSQL `ts_rank` on a GIN-indexed tsvector, top 50 candidates.
  Title and author fields are weighted higher than body text (`setweight 'A'` vs `'B'`).

The two result lists are combined using Reciprocal Rank Fusion (RRF):

```
rrf_score = 1/(60 + vector_rank) + 1/(60 + text_rank)
```

Chunks that appear in both lists get contributions from both terms; chunks in only one list get
a single contribution.

**AND-then-OR fallback for FTS:** The full-text query starts with AND logic
(`plainto_tsquery`, which ANDs all terms). If fewer than 10 chunks match, it falls back to OR
(replacing `&` with `|`). This prevents empty results on compound queries while still preferring
stricter matches when available.

**Why hybrid:** Vector search excels at semantic similarity — finding conceptually related
passages even without shared words. Full-text search excels at exact matches — names, dates,
place names. A query like "Swettenham 1875 Singapore" benefits from both.

**Why RRF constant 60:** Standard value from the RRF literature. Prevents the score from blowing
up for rank-1 results while still rewarding items that rank highly in both systems.

### D2 — Three-stage retrieval: search → rerank → top-k

**What we chose:** Retrieval fetches `rerank_candidates` (default 30) chunks via hybrid search,
then a reranker scores and re-orders them, returning the final `retrieval_top_k` (default 15).

**Reranking:** The reranker prepends document metadata (author, title) to each chunk's text
before scoring, giving the model richer context for relevance judgement. The Bedrock reranker
(Amazon Rerank v1) is used in production.

**Why rerank:** Hybrid search produces a good candidate set but RRF scores are noisy — a chunk
that ranks #1 in vector search and #50 in FTS gets the same RRF score as one that ranks #25 in
both. A cross-encoder reranker can make a more nuanced relevance judgement by reading the full
query-document pair.

### D3 — Agentic tool-calling loop

**What we chose:** After the initial retrieval, the LLM can call two tools:

- **`search_documents`** — search the indexed collection with a different query
- **`find_images`** — search for figures, plates, maps, and photographs by caption
- **`web_search`** — search the web via DuckDuckGo (for general knowledge outside the collection)

The loop runs up to 5 rounds (`MAX_TOOL_ROUNDS`). Each round:
1. Call the LLM with tools enabled (non-streaming)
2. If the LLM returns tool calls, execute them and append results to the conversation
3. Deduplicate new chunks against already-retrieved ones
4. If the context window is nearly full, skip to the final streaming response

**Why agentic:** A single retrieval pass may miss relevant information if the user's phrasing
doesn't match the indexed text. The LLM can reformulate queries (synonyms, broader scope,
different names) and iteratively gather context before synthesising a response.

### D4 — Citation renumbering

**What we chose:** The LLM cites context passages using `[N]` markers where N corresponds to the
passage number in the prompt. After the response is complete, citations are renumbered
consecutively in order of first appearance: `[7]`, `[3]`, `[12]` become `[1]`, `[2]`, `[3]`.

Both single `[N]` and multi-citation `[N, M]` patterns are supported. Multi-citations like
`[23, 24]` are renumbered element-wise: if 23→`[2]` and 24→`[3]`, the result is `[2, 3]`.

A sentinel character (`\x00`) is used during replacement to prevent cascading substitutions
(e.g. `[1]` → `[2]` → `[3]`).

Only chunks that the LLM actually cited are included in the formatted source list. If no
citations are detected (e.g. the LLM forgot to cite), all chunks are shown as a fallback.
Citations inside `<think>` blocks are excluded — only citations in the visible response count.

**Duplicate-source prevention:** The system prompt instructs the LLM not to generate its own
Sources/References section. As a safety net, `strip_llm_sources()` detects and removes any
LLM-generated bibliography before the code appends its own formatted sources. In the streaming
path, if an LLM-generated sources section was already sent, the code skips appending a second one.

**Duplicate-index collapsing:** When multiple context passages come from the same underlying chunk
(same `chunk_id` + `start_page`), `collapse_duplicate_indices()` merges their in-text `[N]`
markers to the same display number. This prevents the confusing case where `[1]`, `[2]`, `[3]`
appear in text but Sources shows only 2 entries because two passages were from the same chunk.

**Source format:**
```
---
**Sources:**
[1] Author (Year), "Title", pp.X-Y — Section Heading
[2] Author (Year), "Title", pp.X-Y
```

**Why renumber:** The LLM may use chunks from multiple tool-calling rounds, producing gaps and
out-of-order numbers in the raw citations. Renumbering gives the reader a clean `[1], [2], [3]`
sequence that matches the source list at the bottom.

### D5 — Context window management

**What we chose:** Dynamic token budgeting to maximise context without exceeding the model's
window:

1. **Initial trim:** After building the prompt with initial chunks, if the input exceeds the
   budget, drop the lowest-relevance chunk and recount. Repeat until it fits.
2. **Tool-round budget check:** Before each tool round, check whether there's enough space for
   another round. If not, skip to the final answer.
3. **Dynamic max_tokens:** `min(llm_max_tokens + thinking_budget, remaining_context) - 64`
4. **Overflow cleanup:** If the context is still too full for a response after tool rounds, drop
   the oldest assistant+tool message groups until it fits.
5. **Minimum output reservation:** 256 tokens (`MIN_OUTPUT_TOKENS`) are always reserved for the
   completion.

**Why:** The context window is shared between input (system prompt + context + history + tool
results) and output (thinking + response). Without dynamic management, long conversations or
many tool rounds would silently truncate the response or cause API errors.

### D6 — System prompt design

**What we chose:** The system prompt instructs the LLM to:

- Write narrative prose that synthesises across sources (not bullet lists)
- Cite every factual claim with `[N]` markers
- Preserve original spellings, names, and dates exactly
- Treat all passage types equally (journal entries, footnotes, abstracts, biographical notes)
- Use `search_documents` for alternative queries if initial context is insufficient
- Use `web_search` for general knowledge outside the collection
- Never invent facts

**Transitive citation guidance:** A guideline instructs the LLM to handle `[cites:]`-marked
footnotes by distinguishing between secondary author claims and primary source records. For
example, when a footnote cites `CO 273/105`, the LLM should say "according to a colonial office
dispatch (cited in Author [N])" rather than attributing the claim directly to the author.

**What we avoided:** Explicit refusal templates ("I cannot answer..."). Qwen3 over-triggers
refusal when the prompt includes them, refusing benign historical queries. Instead, the prompt
says "do not invent facts" and "only state what the sources say", which achieves the same goal
without triggering false refusals.

### D7 — Extended thinking

**What we chose:** A 2048-token thinking budget for Bedrock (Qwen3-235B). Thinking tokens are
enclosed in `<think>...</think>` blocks, stripped from the displayed response, and do not count
toward citations.

Thinking tokens count against `max_tokens` on Bedrock, so the budget is added to `llm_max_tokens`
when computing the dynamic maximum: `max_tokens = llm_max_tokens + thinking_budget`.

**Why 2048:** Large enough for the model to reason through multi-source synthesis without
consuming too much of the context window. Higher budgets produce diminishing returns and slow
responses.

### D8 — Temperature 0.5

**What we chose:** Default temperature 0.5 for all chat completions. Configurable per-request via
the OpenAI-compatible API.

**Why 0.5:** Lower temperatures (0.0–0.3) produce repetitive, mechanical prose. Higher
temperatures (0.7+) introduce hallucination risk in citation-heavy responses. 0.5 gives natural
narrative flow while keeping factual claims grounded in the retrieved context.

### D9 — Lambda non-streaming

**What we chose:** On Lambda, the API forces non-streaming responses (returns `application/json`
instead of SSE), detected by checking for the `AWS_LAMBDA_FUNCTION_NAME` environment variable.

**Why:** Lambda buffered invoke mode adds a `Content-Length` header to SSE responses. Open WebUI
receives the full response body at once with `Content-Type: text/event-stream`, then forwards
the raw SSE text (including `data:` prefixes and `[DONE]` markers) to the browser as visible
content. Forcing `application/json` makes Open WebUI handle the response correctly.

Locally, streaming works normally via SSE.

### D10 — AWS Bedrock for all model inference

**What we chose:** All three model components use AWS Bedrock:

| Component | Bedrock Model |
|-----------|---------------|
| Chat LLM | Qwen3-235B-A22B via Converse API |
| Embedding | Amazon Titan Embed Text v2 (1024 dims) |
| Reranking | Amazon Rerank v1 (via bedrock-agent-runtime) |

The provider abstraction (`LLMProvider`, `EmbedProvider`, `RerankProvider` base classes) is
retained for testability, but only the Bedrock implementations exist. Earlier versions had local
providers (vLLM, sentence-transformers, Qwen3-Reranker) for GPU-equipped development machines;
these were removed since all development now uses Bedrock via `AWS_PROFILE`.

### D11 — OpenAI-compatible API

**What we chose:** A FastAPI server exposing these endpoints:

| Endpoint | What it does |
|----------|-------------|
| `GET /` | Health check |
| `GET /v1/models` | List available models (returns `swetbot`) |
| `POST /v1/chat/completions` | Chat completions (streaming or buffered) |
| `GET /v1/images/{figure_id}` | Serve figure image (local file or S3 redirect) |
| `POST /v1/audio/transcriptions` | Speech-to-text (AWS Transcribe) |
| `POST /v1/audio/speech` | Text-to-speech (Amazon Polly) |

**Authentication:** Optional Bearer token via `CHAT_API_KEY`. If set, all endpoints require
`Authorization: Bearer <token>`.

**Why OpenAI-compatible:** Open WebUI (and many other chat frontends) support the OpenAI API
format natively. By implementing the same interface, we can plug the RAG API into Open WebUI
without any custom integration code.

### D12 — Audio endpoints

**Speech-to-text (Transcribe):**
- Uploads audio to a temp S3 location
- Starts an async AWS Transcribe job with language auto-detection
- Supports a custom vocabulary for domain-specific terms (JMBRAS proper names)
- Polls until complete (timeout 60s), then cleans up temp files

**Text-to-speech (Polly):**
- Maps OpenAI voice names to Amazon Polly voices (e.g. `alloy` → `Arthur`, `nova` → `Amy`)
- Returns MP3 audio stream

### D13 — Image retrieval and serving

**What we chose:** Images are surfaced in two ways:

1. **Contextual** — figures on the same pages as retrieved text chunks are automatically included
   in the context with markdown image syntax, so the LLM can reference them in its response.
2. **Explicit** — the `find_images` tool lets the LLM search figures by caption using hybrid
   search (vector + FTS on the `figures` table), returning top 5 results with markdown URLs.

**Serving:** `GET /v1/images/{figure_id}` serves images directly (no auth required — the
presigned S3 URL is already time-limited):
- **Local mode:** returns the JPEG file from `data/out/{doc_id}/assets/`
- **Lambda mode:** generates an S3 presigned URL (1h expiry) and returns a 302 redirect

The `?thumb=true` query parameter serves the thumbnail instead.

**Absolute URLs:** Image URLs in the LLM context use absolute URLs (`CHAT_API_BASE_URL` +
`/v1/images/{id}`) so that Open WebUI (running on a different domain) can render `<img>` tags.
Browser `<img>` tags cannot send `Authorization: Bearer` headers, which is why the image
endpoint is unauthenticated.

**Figure filtering:** Contextual figures with empty or generic captions (e.g. "Figure on p.30")
are excluded from the LLM context. In local mode, figures whose asset files don't exist on disk
are also filtered out to prevent 404s. The image annotation no longer says "(include in response)"
— the LLM decides which images are relevant based on the system prompt guidance to briefly
introduce each image and skip generic ones.

**Why no VL descriptions:** Captions extracted by docproc are sufficient for search. Adding
VL-generated descriptions would increase processing cost and latency without clear retrieval
benefit for this collection (most figures have explicit captions).

### D14 — Context passage formatting

**What we chose:** Retrieved chunks are formatted with citation-style headers that include a
document type label when available:

```
[1] [Primary Source] Swettenham (1875), "Journal 1874-1876", pp.12-14 — October Diary
[1] [Journal Article] Abdullah (2011), "Title", pp.1-22 — Introduction
```

The system prompt includes guidance: primary sources are authoritative for factual claims about
historical events; journal articles provide scholarly interpretation and broader context.

The formatted citation output also includes the document type as a suffix (e.g. `· Primary Source`).

**Metadata fallback:** If the database `author` or `year` fields are empty (common for
historical documents where metadata was not parsed), the system parses the source filename
(e.g. `Abdullah (2011) JMBRAS 84(1), 1-22.pdf` → author: `Abdullah`, year: `2011`).

**Why citation-style headers:** The LLM needs to know which document and page a passage comes
from to produce accurate citations. The header format matches academic citation conventions,
which the LLM understands well.

**Why document type labels:** The JMBRAS corpus mixes primary sources (diaries, letters) and
secondary sources (journal articles). Tagging each passage lets the LLM weigh evidence
appropriately — preferring primary sources for factual claims and journal articles for analysis.

### D15 — Structured metadata filtering

**What we chose:** Both `search_documents` and `find_images` tools accept optional metadata
filters that map to SQL WHERE clauses on the `documents` table:

| Parameter | Type | Applies to | Description |
|-----------|------|-----------|-------------|
| `document_type` | string | search, images | Filter by document type (existing) |
| `year_from` | integer | search, images | Publication year lower bound (inclusive) |
| `year_to` | integer | search, images | Publication year upper bound (inclusive) |
| `language` | string | search | ISO language code: en, ms, zh, ar |
| `publication` | string | search | Exact publication name (e.g. JMBRAS, JSBRAS) |

All filters use the `IS NULL OR` pattern — a NULL parameter means "no filter", so unfiltered
queries are unaffected. Documents with NULL year are excluded when a year range is specified.

The system prompt guides the LLM to map natural language to these filters (e.g. "1870s" →
year_from=1870, year_to=1879; "Malay-language" → language=ms).

**Why:** Questions like "find articles from the 1870s" or "Malay-language sources about
Swettenham" previously relied on keyword matching. Structured SQL filters give precise results
without depending on year/language appearing in the text.

**No DB changes needed:** The `year`, `language`, and `publication` columns already exist in the
`documents` table, populated by the docproc pipeline.

## Configuration

Key environment variables (prefix `CHAT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_LLM_TEMPERATURE` | `0.5` | LLM temperature |
| `CHAT_LLM_MAX_TOKENS` | `4096` | Max output tokens (excluding thinking) |
| `CHAT_LLM_CONTEXT_WINDOW` | `40960` | Total context window size |
| `CHAT_LLM_THINKING_BUDGET` | `2048` | Extended thinking token budget |
| `CHAT_RETRIEVAL_TOP_K` | `15` | Final number of chunks after reranking |
| `CHAT_RERANK_ENABLED` | `true` | Enable reranking stage |
| `CHAT_RERANK_CANDIDATES` | `30` | Chunks to fetch before reranking |
| `CHAT_EMBED_TASK_PREFIX` | `""` | Query embedding prefix |
| `CHAT_EMBED_DIMENSIONS` | `1024` | Embedding vector dimensions |
| `CHAT_WEB_SEARCH_ENABLED` | `true` | Include web_search tool |
| `CHAT_BEDROCK_REGION` | `eu-west-2` | AWS region for Bedrock |
| `CHAT_BEDROCK_CHAT_MODEL_ID` | `qwen.qwen3-235b-a22b-2507-v1:0` | Bedrock chat model |
| `CHAT_BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `CHAT_BEDROCK_RERANK_REGION` | `eu-central-1` | AWS region for reranking |
| `CHAT_BEDROCK_RERANK_MODEL_ID` | `amazon.rerank-v1:0` | Bedrock rerank model |
| `CHAT_DATABASE_DSN` | _(empty)_ | Override full connection string (e.g. Neon DSN) |
| `CHAT_API_PORT` | `8000` | API server port |
| `CHAT_API_KEY` | _(empty)_ | Bearer token (if set, requires auth) |
| `CHAT_S3_BUCKET` | _(empty)_ | S3 bucket for image assets (Lambda mode) |
| `CHAT_API_BASE_URL` | `http://localhost:8000` | Base URL for absolute image URLs in LLM context |
| `CHAT_DATA_DIR` | `data/out` | Local asset root for image serving |
| `CHAT_TRANSCRIBE_S3_BUCKET` | _(empty)_ | S3 bucket for temp transcription files |
| `CHAT_TRANSCRIBE_VOCABULARY_NAME` | _(empty)_ | AWS Transcribe custom vocabulary |

See root `CLAUDE.md` for all CLI commands.

## Deployment

The RAG API runs as a Lambda function behind API Gateway, using Lambda Web Adapter to proxy HTTP
requests to the FastAPI server. See `DEPLOYMENT.md` for Terraform and CI/CD details.

**Key Lambda considerations:**
- Non-streaming forced on Lambda (see D9)
- Cold start ~14s (FastAPI/uvicorn), continues after Lambda's 10s init timeout
- `[cloud]` optional dependency provides `boto3` for Bedrock, Polly, Transcribe, S3
