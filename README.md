# raskl-rag

A pipeline for processing historical JMBRAS (Journal of the Malayan Branch of the Royal Asiatic
Society) and Swettenham PDFs into a searchable RAG (Retrieval-Augmented Generation) system with a
Gradio chat interface.

## Repository structure

```
apps/
  docproc/          PDF → structured JSONL (Qwen3 VL via Bedrock)
  chunker_indexer/  JSONL → chunks → embeddings → PostgreSQL/pgvector
  chat_ui/          Gradio chat interface with agentic RAG search
infra/              Terraform for AWS serverless deployment
docker/             Dockerfiles for Lambda containers
scripts/            Shell scripts for local development workflows
```

Each app has a README with design decisions and rationale:

- [`apps/docproc/README.md`](apps/docproc/README.md) — PDF processing pipeline
- [`apps/chunker_indexer/README.md`](apps/chunker_indexer/README.md) — chunking, embedding, hybrid search
- [`apps/chat_ui/README.md`](apps/chat_ui/README.md) — retrieval, reranking, chat interface

See [`CLAUDE.md`](CLAUDE.md) for all individual commands.

## Setup

```bash
# Requires Python 3.11+ and uv (https://docs.astral.sh/uv/)
# Requires AWS credentials with Bedrock access
uv sync --all-packages
```

## Local development

Local Gradio UI and PostgreSQL, with all model inference via AWS Bedrock. No GPU needed.

**Step 1 — Process PDFs**

```bash
# Process all PDFs (Qwen3 VL via Bedrock, parallel)
./scripts/docproc.sh

# Single PDF
uv run ras-docproc run --pdf "path/to/file.pdf" --backend qwen3vl
```

**Step 2 — Embed and index**

```bash
./scripts/embed.sh
```

Starts local PostgreSQL, initialises the schema, and indexes all processed documents using
Bedrock Titan Embed v2.

**Step 3 — Start the chat UI**

```bash
./scripts/start-chat.sh
```

Starts local PostgreSQL and the Gradio UI on port 7860. LLM, embedding, and reranking all
run via Bedrock.

---

## Serverless deployment (AWS)

Upload a PDF → DocProc Lambda (Qwen3 VL) → versioned JSONL to S3 → Chunker Lambda
(Bedrock Titan Embed) → Neon pgvector → Chat Lambda (Gradio + Bedrock).

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full details. The serverless stack uses:

- AWS Lambda (chat + docproc + upload) with Bedrock for all model inference
- Neon (serverless PostgreSQL + pgvector)
- S3 for PDF upload and docproc output
- Terraform in `infra/` to provision everything
- GitHub Actions for deploy-on-push and per-PR preview environments

---

## Diagnostic scripts

```bash
# Diagnose RAG pipeline (embedding → retrieval → reranking → LLM) step by step
uv run python scripts/diagnose_rag.py --query "What dates did Swettenham go to Singapore?"

# Migrate local DB chunks to Neon with Bedrock re-embedding
uv run python scripts/migrate_to_neon.py --dry-run
```

## HTML debug reports

After running docproc, inspect extraction results with an interactive HTML report:

```bash
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10
```

## Testing

```bash
# docproc end-to-end tests (uses real PDFs from docs/)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# chunker/indexer tests
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v
```
