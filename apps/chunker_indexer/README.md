# ras-chunker-indexer — Chunking, Embedding and Indexing

## What this app does

`ras-chunker-indexer` takes the JSONL output produced by `ras-docproc` and prepares it for
search. It does three things:

1. **Chunk** — splits document text into passages sized for retrieval (heading-aware, cross-page
   stitching, footnotes inlined)
2. **Embed** — converts each passage to a numeric vector using an embedding model
3. **Index** — stores the text and vectors in PostgreSQL (with the pgvector extension) for hybrid
   search

## How it fits into the pipeline

```
data/out/{doc_id}/   (from ras-docproc)
    │
    ▼
ras-chunker-indexer  ← this app
    │  loads JSONL → restitches paragraphs → splits into chunks
    │  → embeds each chunk → stores in PostgreSQL
    ▼
PostgreSQL + pgvector
    │
    ▼
ras-chat-ui  (retrieves relevant chunks for each user query)
```

## Architecture overview

The pipeline runs in this order for each document:

1. **Load** — reads `documents.jsonl`, `text_blocks.jsonl`, `footnotes.jsonl`,
   `footnote_refs.jsonl` from docproc output
2. **Restitch** — merges paragraphs that were split across page boundaries
3. **Chunk** — splits into retrieval-sized passages (heading-bounded, ≤512 tokens)
4. **Embed** — generates a 1024-dimension vector for each chunk
5. **Index** — upserts into the `documents` and `chunks` tables

## Key design decisions

### D1 — Heading-based semantic chunking, not fixed-size windows

**What we chose:** A new `heading` block always starts a new chunk. Paragraphs accumulate until
adding the next one would exceed 512 tokens, at which point the current chunk is flushed and a
new one begins with the same heading carried forward.

Token counting uses a rough estimate of one token per four characters. This avoids loading a
tokenizer just for chunking and is accurate enough at this scale.

**Why not fixed-size windows:** Splitting mid-sentence or mid-argument degrades retrieval quality.
A chunk that starts mid-sentence is harder to understand in isolation, produces a confusing vector
representation, and looks odd when cited in the chat UI. Headings are natural semantic boundaries
in academic writing: a new heading signals a new topic.

**Why 512 tokens:** The embedding models used (Qwen3-Embedding-8B and BGE-M3) have context windows
in the range of 512–8192 tokens, but embedding quality degrades for very long inputs. 512 tokens
gives a passage long enough to carry meaningful context while fitting comfortably in the model's
effective range. It also means many chunks fit in the LLM's prompt window, so more context is
available per query.

### D2 — Cross-page paragraph restitching

**What we chose:** Before chunking, consecutive paragraphs split across a page boundary are merged
back together. A merge happens when all of the following are true:

- Both blocks are type `paragraph`
- The second block is on the immediately following page
- The second block is the first on its page (reading order 0)
- The first block does not end with sentence-ending punctuation (`.`, `!`, `?`, `"`, `'`, `)`)
- The second block starts with a lowercase letter

**Why:** A PDF page break is a physical artefact, not a semantic break. When a paragraph is split
across two pages, the text in `text_blocks.jsonl` contains two separate records. If these are
chunked separately, the embedding for the first half is incomplete — it represents a sentence that
stops mid-thought. Merging restores the complete semantic unit before embedding.

**Why these criteria:** The sentence-ending punctuation check ensures we don't merge paragraphs
that happen to be adjacent on consecutive pages but are actually separate thoughts. The lowercase-
start check catches the common case where a sentence continues across a page turn (since sentences
in the middle of a paragraph don't start with capitals). Together they give high precision with
very few false merges in practice.

### D3 — Footnotes inlined into chunks

**What we chose:** Footnote text is appended to the chunk that contains the corresponding
in-text reference. The format is:

```
[Heading if present]

[Body paragraph text...]

---
Footnotes:
[1] The full text of footnote 1
[2] The full text of footnote 2
```

**Why:** A footnote about a person or source mentioned in a paragraph is most useful when retrieved
alongside that paragraph. If footnotes were stored as separate chunks, a search for a person cited
only in a footnote would either retrieve just the footnote (no context) or miss it entirely (if
the footnote doesn't contain enough standalone information to score well). Inlining keeps the
context together.

The footnote numbers in the inlined text correspond to the `[ref:N]` markers in the body text,
so the relationship between claim and footnote is preserved in the chunk text itself.

### D4 — Hybrid search: vector + full-text with Reciprocal Rank Fusion

**What we chose:** Two independent searches run in parallel on each query:

- **Vector search:** cosine similarity between the query embedding and all stored chunk embeddings,
  top 50 candidates
- **Full-text search:** PostgreSQL `ts_rank` on a GIN-indexed tsvector, using OR logic, top 50
  candidates (title and author fields weighted higher than body text)

The two result lists are combined using Reciprocal Rank Fusion (RRF):

```
rrf_score = 1/(60 + vector_rank) + 1/(60 + text_rank)
```

Chunks that appear in both lists get contributions from both terms; chunks that appear in only one
list get a single contribution.

**Why hybrid:** Vector search excels at semantic similarity — it finds passages that are
conceptually related to the query even if they don't share exact words. Full-text search excels at
exact matches — names, dates, place names, and other proper nouns that might not translate well to
vector space. A query like "Swettenham 1875 Singapore" benefits from both: vector search finds
contextually relevant passages, FTS finds passages that contain those specific terms.

**Why OR logic (not AND):** With AND logic, a query like "Swettenham 1875 Singapore" would require
all three terms to appear in the same passage. A journal entry from 1875 that discusses Singapore
but attributes authorship to Swettenham only in a separate header block would be missed. OR logic
matches any passage containing any query term, letting the RRF scoring surface passages with
multiple matches above those with only one.

**Why RRF constant 60:** This is a standard value from the RRF literature. The constant prevents
the score from blowing up for rank-1 results (which would be `1/1 = 1.0` without the constant)
while still rewarding items that rank highly in both systems. At constant 60, a result ranked #1
in both systems scores `1/61 + 1/61 ≈ 0.033`, while a result ranked #50 in both scores only
`1/110 + 1/110 ≈ 0.018`.

### D5 — HNSW index for approximate vector search

**What we chose:**

```sql
CREATE INDEX idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbour index.
It builds a multi-layer graph where each node connects to its `m=16` nearest neighbours. At query
time, it navigates this graph to find approximate nearest neighbours very quickly.

`m=16` is the number of connections per node (higher = better recall, larger index).
`ef_construction=200` is the size of the candidate list during index construction (higher = better
index quality at the cost of slower builds).

**Why HNSW over IVFFlat:** pgvector supports two index types. IVFFlat requires you to pre-specify
the number of clusters (which depends on the dataset size) and needs to be rebuilt as the collection
grows. HNSW doesn't require pre-specification and maintains good recall across a wide range of
collection sizes. For a collection of this scale (thousands to tens of thousands of chunks), HNSW
is the simpler, better-performing choice.

### D6 — Task-specific embedding prefixes

**What we chose:** Document chunks are embedded with:
```
"Represent this document passage for retrieval: "
```

Query strings are embedded with:
```
"Represent this query for retrieving relevant passages: "
```

**Why:** The Qwen3-Embedding-8B model is trained to distinguish between document-type and
query-type inputs. The prefix tells the model which mode to use, and the model encodes documents
and queries into compatible but non-identical vector spaces optimised for retrieval. Omitting the
prefix or using the wrong prefix measurably reduces retrieval quality with this model family.

**Bedrock Titan Embed v2:** Task prefixes are set to empty string (`""`) when using Bedrock. The
chunker's `CHUNKER_EMBED_TASK_PREFIX` must match whatever prefix was used when the collection was
indexed — mixing prefixes between indexing and query time will produce poor results.

### D7 — Page offset stored but not re-applied

**What we chose:** The `documents` table has a `page_offset` column that stores the offset
computed by docproc. It is loaded from `documents.jsonl` and stored as-is. The `chunks` table
stores `start_page` and `end_page` directly — these are already the correct journal page numbers
(offset was applied during docproc Stage 4).

**Why this matters:** An earlier version of the code added the offset again in the chat UI when
displaying citations, producing double-offset page numbers. For example, an article on journal
pages 76–100 in a 151-page PDF (offset = −51) would display page 25 instead of page 76.

The fix: `page_offset` is retained in the database as an audit record (so we know what was
applied), but display code must use `start_page` directly without adding the offset again.

### D8 — Bedrock embedding

**What we chose:** Embedding uses AWS Bedrock (Amazon Titan Embed Text v2) via boto3. Titan
embeds one text per API call; the provider uses a `ThreadPoolExecutor` (10 workers) to embed
chunks concurrently.

Selected via `CHUNKER_EMBED_PROVIDER="bedrock"`.

**Important:** The embedding model used by the chunker must match the model used by the chat UI's
retriever. Both must use the same Bedrock model ID. Mixing models produces silently poor
retrieval because vectors from different models are not comparable.

### D9 — AWS Lambda deployment

**What we chose:** An S3-triggered Lambda handler (`lambda_handler.py`) that listens for
`processed/{doc_id}/v{N}/documents.jsonl` uploads from the docproc Lambda. On trigger, it:

1. Downloads all JSONL files for that version from S3
2. Runs the full chunk + embed + index pipeline
3. Upserts into Neon (the cloud PostgreSQL database)

The version number from the S3 key is passed through to `upsert_chunks()` for audit tracking.

**`[cloud]` dependency:** The Lambda handler requires `boto3`, which is an optional `[cloud]`
extra in `pyproject.toml`.

### D10 — Chunk change auditing

**What we chose:** A `chunk_changes` table that records what changed each time a document is
re-indexed. Each row captures: `doc_id`, `version`, `chunks_total`, `chunks_added`,
`chunks_removed`, `chunks_text_changed`, and `chunks_unchanged`.

**Why:** When re-processing a PDF (e.g. after improving the OCR pipeline), it's useful to know
how many chunks actually changed. This audit trail helps measure whether pipeline improvements
are producing different (hopefully better) output, and by how much.

## Database schema

```sql
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    title           TEXT,
    author          TEXT,
    year            INTEGER,
    page_offset     INTEGER NOT NULL DEFAULT 0,  -- audit only, already applied
    sha256_pdf      TEXT NOT NULL,
    indexed_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    start_page      INTEGER NOT NULL,  -- journal page number (offset already applied)
    end_page        INTEGER NOT NULL,
    section_heading TEXT,
    text            TEXT NOT NULL,
    block_ids       TEXT[] NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       vector(1024) NOT NULL,
    indexed_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunk_changes (
    id              SERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    version         INTEGER NOT NULL,
    indexed_at      TIMESTAMPTZ DEFAULT now(),
    chunks_total    INTEGER NOT NULL,
    chunks_added    INTEGER NOT NULL DEFAULT 0,
    chunks_removed  INTEGER NOT NULL DEFAULT 0,
    chunks_text_changed INTEGER NOT NULL DEFAULT 0,
    chunks_unchanged INTEGER NOT NULL DEFAULT 0
);
```

## Configuration

Key environment variables (prefix `CHUNKER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKER_EMBED_PROVIDER` | `bedrock` | `"bedrock"` (or `"vllm"` for legacy local use) |
| `CHUNKER_EMBED_TASK_PREFIX` | `""` | Must match chat UI embed prefix |
| `CHUNKER_MAX_CHUNK_TOKENS` | `512` | Maximum tokens per chunk |
| `CHUNKER_RESTITCH_ENABLED` | `true` | Enable cross-page paragraph merging |
| `CHUNKER_DB_NAME` | `raskl_rag` | PostgreSQL database name |
| `CHUNKER_DATABASE_DSN` | _(empty)_ | Override full connection string (e.g. Neon DSN) |
| `CHUNKER_BEDROCK_REGION` | `eu-west-2` | AWS region for Bedrock (bedrock provider only) |
| `CHUNKER_BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model ID |

See root `CLAUDE.md` for all CLI commands.

## Testing

```bash
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v
```

Key test files:

| File | What it tests |
|------|---------------|
| `test_chunker.py` | Heading-based splitting, token limits, footnote inlining |
| `test_restitch.py` | Cross-page merge criteria (all five conditions) |
