# raskl-rag

Monorepo for processing historical JMBRAS/Swettenham PDFs into structured JSONL for RAG indexing.

## Structure

- `apps/docproc/` — Main document processing pipeline (`ras-docproc`)
- `apps/chunker_indexer/` — Chunk + index JSONL into PostgreSQL/pgvector (`ras-chunker`)
- `apps/vllm_launcher/` — Launch and manage vLLM model servers, download models (`ras-vllm-launcher`)
- `apps/chat_ui/` — Gradio chat interface with agentic RAG (`ras-chat-ui`)
- `infra/` — Terraform for AWS serverless deployment (Lambda, Neon, S3, ECR)
- `docker/` — Dockerfiles for Lambda containers (`chat/`, `docproc/`)
- `.github/workflows/` — CI/CD (deploy on push to master, PR chat previews)

## Commands

```bash
# Install all workspace packages
uv sync --all-packages

# Run docproc pipeline on a PDF (Docling backend, default for clean PDFs)
uv run ras-docproc run --pdf "docs/clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf" --max-pages 5

# Run with DeepSeek-OCR backend (auto-detected for messy PDFs, requires vLLM server)
uv run ras-docproc run --pdf "docs/messy/Swettenham Journal 1874-1876.pdf" --page-range 76

# Explicitly choose backend
uv run ras-docproc run --pdf path/to/file.pdf --backend deepseek

# Start/stop DeepSeek-OCR vLLM server
uv sync --package ras-vllm-launcher --extra gpu
uv run ras-vllm-launcher ocr        # start (waits until ready)
uv run ras-vllm-launcher health     # check status
uv run ras-vllm-launcher down       # stop

# Generate HTML debug report
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10

# Run tests (e2e-focused)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# --- Model Downloads ---

# Download Qwen3 embedding + reranker models to ./models/
uv run ras-vllm-launcher download --role all

# --- Chunker/Indexer ---

# Start PostgreSQL (pgvector)
docker compose up -d

# Initialize database schema
uv run ras-chunker init-db

# Dry-run: show chunk plan without embedding/indexing
uv run ras-chunker plan --doc-id DOC_ID

# Index a document (requires vLLM embedding server + PostgreSQL)
uv run ras-chunker index --doc-id DOC_ID

# Index all processed documents
uv run ras-chunker index-all

# Run chunker tests
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v
```

## Architecture

The `docproc` pipeline processes PDFs through these stages:
1. **Inventory** — discover PDF, compute SHA256, generate doc_id
2. **Extract (Docling or DeepSeek)** — structured content extraction (Docling for clean PDFs, DeepSeek-OCR via vLLM for scanned/messy PDFs)
3. **Extract (MuPDF)** — low-level text/image/font extraction via PyMuPDF
4. **Normalize** — NFKC normalization, dehyphenation, text cleaning
5. **Boilerplate** — detect/remove headers, footers, repeated lines
6. **Language** — detect language per block (en, ms, zh, ar)
7. **Footnotes** — detect footnote blocks in page footer zone
8. **Footnote refs** — link in-body superscript/bracket refs to footnotes
9. **Figures** — extract embedded images, generate JPG + thumbnails
10. **Captions** — match "Fig. N" / "Plate N" text to figures
11. **Rotation** — detect vertical text, suggest page rotation
12. **Export** — write JSONL files + assets to `data/out/{doc_id}/`

## Provider Abstraction

Both `chat_ui` and `chunker_indexer` use a provider pattern for model inference, allowing transparent switching between local (vLLM/sentence-transformers) and cloud (AWS Bedrock) backends via environment variables.

### chat_ui providers (`apps/chat_ui/src/ras_chat_ui/providers/`)

| Provider type | Local backend | Cloud backend |
|---------------|---------------|---------------|
| LLM | `vllm` — httpx to vLLM OpenAI-compatible API | `bedrock` — Converse API |
| Embedding | `sentence-transformers` — SentenceTransformer | `bedrock` — Cohere Embed Multilingual v3 |
| Reranking | `qwen3` / `cross-encoder` — local models | `bedrock` — Cohere Rerank 3.5 |

Config fields: `CHAT_LLM_PROVIDER`, `CHAT_EMBED_PROVIDER`, `CHAT_RERANK_PROVIDER`

### chunker_indexer providers (`apps/chunker_indexer/src/ras_chunker/providers/`)

| Provider type | Local backend | Cloud backend |
|---------------|---------------|---------------|
| Embedding | `vllm` — httpx to vLLM embeddings API | `bedrock` — Cohere Embed Multilingual v3 |

Config field: `CHUNKER_EMBED_PROVIDER`

## AWS Serverless Deployment

See `DEPLOYMENT.md` for full details. Summary:

- **Terraform** (`infra/`) provisions Neon (pgvector), S3, ECR, Lambda functions, IAM
- **Chat Lambda** — Gradio via Lambda Web Adapter, Bedrock for LLM/embed/rerank
- **DocProc Lambda** — S3-triggered, Docling + chunk + Bedrock embed + Neon index
- **Upload Lambda** — HTML form with password auth → S3 presigned upload
- **GitHub Actions** — deploy on push to master, per-PR chat previews

## Documentation

Each app has a README that documents its design decisions and rationale:

- `apps/docproc/README.md` — PDF processing pipeline: backends, page offset, boilerplate, footnotes, IDs
- `apps/chunker_indexer/README.md` — chunking, embedding, hybrid search, HNSW, task prefixes
- `apps/chat_ui/README.md` — retrieval, reranking, citation renumbering, grounding, provider architecture

**When making any architectural change** (new feature, changed algorithm, tuned parameter, added
model, changed prompt), update the relevant README to record:
- What changed
- Why it was changed (what problem it solved or what was observed)
- Any alternatives considered

## Coding Conventions

- Python 3.11+
- Pydantic v2 for all data models
- ruff for formatting (line-length=120)
- Click for CLIs
- E2E tests preferred over unit tests
- Tests use actual PDFs from `docs/` directory
- Provider pattern for swappable model backends (local vs cloud)
- `boto3` is an optional `[cloud]` dependency in chat_ui and chunker_indexer
- Update the relevant app README when making architectural changes (see Documentation section)
