# SwetBot

A RAG system for exploring historical journals of the Malayan Branch of the Royal Asiatic Society
(JMBRAS). It processes scanned and born-digital PDFs into searchable, citable passages and serves
them through an Open WebUI chat interface backed by an agentic retrieval pipeline.

## What is this?

The JMBRAS journals span 150 years (1870s–present) and cover the history, ethnography, and natural
history of Southeast Asia. Many exist only as scanned PDFs with inconsistent OCR, mixed languages
(English, Malay, Chinese, Arabic), and complex layouts with footnotes, plates, and maps.

SwetBot turns these PDFs into a conversational research tool: ask a question, get a narrative
answer with page-level citations back to the original journals.

## Architecture

```
PDF files (scanned or born-digital)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ras-docproc                                    │
│  Qwen3 VL (vision-language model) extracts      │
│  structured text, footnotes, figures, metadata  │
└─────────────────────┬───────────────────────────┘
                      │  JSONL files + image assets
                      ▼
┌─────────────────────────────────────────────────┐
│  ras-chunker-indexer                            │
│  Heading-aware chunking → Cohere Embed v4        │
│  → PostgreSQL/pgvector (hybrid index)           │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  ras-rag-engine                                 │
│  Hybrid search (vector + FTS + RRF) → rerank   │
│  → agentic tool-calling loop → cited response   │
└─────────────────────┬───────────────────────────┘
                      │  OpenAI-compatible API
                      ▼
               ┌──────────────┐
               │  Open WebUI  │
               │  Chat UI     │
               └──────────────┘
```

## Key features

- **Vision-language extraction** — Qwen3 VL reads each page as an image, handling messy 19th-century
  scans and clean modern PDFs with the same pipeline
- **Footnote detection and linking** — footnotes are detected, linked to their in-text references,
  and inlined into the chunks that cite them
- **Figure and plate search** — embedded images are extracted, captioned, and searchable via a
  dedicated `find_images` tool
- **Hybrid search** — vector similarity + full-text search combined with Reciprocal Rank Fusion,
  then cross-encoder reranking
- **Agentic RAG** — the LLM can call `search_documents`, `find_images`, and `web_search` to
  iteratively gather context before answering (up to 5 tool rounds)
- **Citation renumbering** — `[N]` markers are renumbered consecutively and a formatted source list
  is appended to each response
- **Extended thinking** — 2048-token thinking budget for multi-source reasoning (hidden from output)
- **Page offset correction** — JSTOR/MUSE cover pages are detected and page numbers corrected so
  citations match the printed journal
- **Multi-language support** — per-block language detection for English, Malay, Chinese, and Arabic

## Repository structure

| Directory | Package | What it does |
|-----------|---------|-------------|
| `apps/docproc/` | `ras-docproc` | PDF → structured JSONL (16-stage pipeline using Qwen3 VL) |
| `apps/chunker_indexer/` | `ras-chunker` | JSONL → heading-aware chunks → embeddings → PostgreSQL/pgvector |
| `apps/rag_engine/` | `ras-rag-engine` | OpenAI-compatible RAG API (FastAPI, hybrid search, agentic tool use) |
| `infra/` | — | Terraform for AWS serverless deployment |
| `scripts/` | — | Shell scripts for local development workflows |

Each app has a detailed README documenting design decisions and rationale:

- [`apps/docproc/README.md`](apps/docproc/README.md) — extraction pipeline, boilerplate detection, footnote linking
- [`apps/chunker_indexer/README.md`](apps/chunker_indexer/README.md) — chunking strategy, restitching, embedding, schema
- [`apps/rag_engine/README.md`](apps/rag_engine/README.md) — hybrid search, reranking, tool-calling loop, citations

## Prerequisites

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **AWS credentials** with access to Bedrock models (Qwen3 VL, Cohere Embed v4, Cohere Rerank 3.5)
- **Docker** (for PostgreSQL/pgvector and Open WebUI)
- No GPU needed — all inference runs on AWS Bedrock

## Models

All model inference uses AWS Bedrock. No local model serving required.

| Component | Bedrock Model |
|-----------|---------------|
| OCR / extraction | Qwen3-VL-235B via Converse API |
| Chat LLM | Qwen3-235B-A22B via Converse API |
| Embedding | Cohere Embed v4 via EU inference profile (1024 dims) |
| Reranking | Cohere Rerank 3.5 |

## Quick start (local)

```bash
# Install all workspace packages
uv sync --all-packages
```

**Step 1 — Process PDFs** — extract text, footnotes, figures, and metadata from each page using
Qwen3 VL:

```bash
./scripts/docproc.sh            # all PDFs in docs/, parallel
uv run ras-docproc run --pdf "path/to/file.pdf"  # single PDF
```

**Step 2 — Embed and index** — start local PostgreSQL, chunk the extracted text, embed with Titan
v2, and index into pgvector:

```bash
./scripts/embed.sh
```

**Step 3 — Start the RAG API** — OpenAI-compatible API on port 8000:

```bash
./scripts/start-api.sh
```

**Step 4 — Start Open WebUI** — chat interface on port 3000, connected to the RAG API:

```bash
docker compose up open-webui
```

## Serverless deployment (AWS)

The full pipeline runs serverlessly on AWS:

```
Upload PDF → DocProc Lambda → versioned JSONL to S3
  → Chunker Lambda → Neon pgvector → RAG API Lambda → Open WebUI (App Runner)
```

Stack: Lambda (RAG API + docproc + upload), App Runner (Open WebUI), Neon (serverless PostgreSQL),
S3, Bedrock, Terraform in `infra/`, GitHub Actions for deploy-on-push.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full details.

## Testing

```bash
# docproc (e2e tests using real PDFs from docs/)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# chunker/indexer
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v

# retrieval benchmark (requires indexed database + Bedrock credentials via .env)
uv run pytest tests/e2e/ -v          # local PostgreSQL
uv run pytest tests/e2e/ -v --live   # Neon
```

## HTML debug reports

Inspect extraction results with an interactive HTML report:

```bash
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10
```

## Further reading

- [`CLAUDE.md`](CLAUDE.md) — full command reference and coding conventions
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — AWS serverless deployment guide
