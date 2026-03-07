# raskl-rag

Monorepo for processing historical JMBRAS/Swettenham PDFs into structured JSONL for RAG indexing.

## Structure

- `apps/docproc/` — Main document processing pipeline (`ras-docproc`)
- `apps/chunker_indexer/` — Chunk + index JSONL into PostgreSQL/pgvector (`ras-chunker`)
- `apps/chat_ui/` — Gradio chat interface with agentic RAG (`ras-chat-ui`)
- `infra/` — Terraform for AWS serverless deployment (Lambda, Neon, S3, ECR)
- `docker/` — Dockerfiles for Lambda containers (`chat/`, `docproc/`)
- `.github/workflows/` — CI/CD (deploy on push to master, PR chat previews)

## Two modes of operation

### Local mode
Local Gradio UI, local PostgreSQL, local PDF files. All model inference via AWS Bedrock (`AWS_PROFILE=linusnorton`).

```bash
# 1. Process all PDFs (Qwen3 VL via Bedrock)
./scripts/docproc.sh

# 2. Embed + index into local PostgreSQL (Bedrock Titan Embed v2)
./scripts/embed.sh

# 3. Start chat UI (Bedrock for LLM/embed/rerank)
./scripts/start-chat.sh
```

### Deployed mode
Everything in AWS: Lambda, Neon (serverless PostgreSQL), S3, Bedrock.

Upload a PDF → DocProc Lambda (Qwen3 VL) → versioned JSONL to S3 → Chunker Lambda
(Bedrock Titan Embed) → Neon pgvector → Chat Lambda (Gradio + Bedrock).

See `DEPLOYMENT.md` for full details.

## Commands

```bash
# Install all workspace packages
uv sync --all-packages

# Run docproc on a single PDF (Qwen3 VL via Bedrock)
uv run ras-docproc run --pdf "path/to/file.pdf" --backend qwen3vl

# Run docproc on all PDFs in docs/
uv run ras-docproc run-all --docs-dir docs --backend qwen3vl

# Generate HTML debug report
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10

# Run docproc tests (e2e-focused)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# --- Chunker/Indexer ---

# Start PostgreSQL (pgvector)
docker compose up -d

# Initialize database schema
uv run ras-chunker init-db

# Dry-run: show chunk plan without embedding/indexing
uv run ras-chunker plan --doc-id DOC_ID

# Index a document (Bedrock Titan Embed v2 + PostgreSQL)
CHUNKER_EMBED_PROVIDER=bedrock uv run ras-chunker index --doc-id DOC_ID

# Index all processed documents
CHUNKER_EMBED_PROVIDER=bedrock uv run ras-chunker index-all

# Run chunker tests
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v
```

## Architecture

The `docproc` pipeline processes PDFs through these stages:
1. **Inventory** — discover PDF, compute SHA256, generate doc_id
2. **Extract (Qwen3 VL)** — structured content extraction via Bedrock (or Docling/DeepSeek for legacy local use)
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

All model inference uses AWS Bedrock:

| Component | Bedrock Model |
|-----------|---------------|
| Chat LLM | Qwen3-235B-A22B via Converse API |
| Embedding | Amazon Titan Embed Text v2 (1024 dims) |
| Reranking | Amazon Rerank v1 |
| OCR (docproc) | Qwen3-VL-235B via Converse API |

Config fields: `CHAT_LLM_PROVIDER`, `CHAT_EMBED_PROVIDER`, `CHAT_RERANK_PROVIDER`, `CHUNKER_EMBED_PROVIDER`

## AWS Serverless Deployment

See `DEPLOYMENT.md` for full details. Summary:

- **Terraform** (`infra/`) provisions Neon (pgvector), S3, ECR, Lambda functions, IAM
- **Chat Lambda** — Gradio via Lambda Web Adapter, Bedrock for LLM/embed/rerank
- **DocProc Lambda** — S3-triggered, Qwen3 VL + chunk + Bedrock embed + Neon index
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
- Provider pattern for swappable model backends (Bedrock default, local alternatives)
- `boto3` is an optional `[cloud]` dependency in docproc, chat_ui and chunker_indexer
- Update the relevant app README when making architectural changes (see Documentation section)
