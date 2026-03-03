# raskl-rag

A pipeline for processing historical JMBRAS (Journal of the Malayan Branch of the Royal Asiatic
Society) and Swettenham PDFs into a searchable RAG (Retrieval-Augmented Generation) system with a
Gradio chat interface.

## Repository structure

```
apps/
  docproc/          PDF → structured JSONL (text, footnotes, figures, metadata)
  chunker_indexer/  JSONL → chunks → embeddings → PostgreSQL/pgvector
  vllm_launcher/    Start/stop vLLM model servers, download models
  chat_ui/          Gradio chat interface with agentic RAG search
infra/              Terraform for AWS serverless deployment
docker/             Dockerfiles for Lambda containers
scripts/            Shell scripts for common workflows
```

Each app has a README with design decisions and rationale:

- [`apps/docproc/README.md`](apps/docproc/README.md) — PDF processing pipeline
- [`apps/chunker_indexer/README.md`](apps/chunker_indexer/README.md) — chunking, embedding, hybrid search
- [`apps/chat_ui/README.md`](apps/chat_ui/README.md) — retrieval, reranking, chat interface

See [`CLAUDE.md`](CLAUDE.md) for all individual commands.

## Setup

```bash
# Requires Python 3.11+ and uv (https://docs.astral.sh/uv/)
uv sync --all-packages
```

## Two local stacks

There are two tested local configurations depending on available GPU memory.

### Heavy stack — RTX 5090 / 32GB VRAM

Chat model: Qwen3-30B-A3B-GPTQ-Int4 (~16GB VRAM). Embedding and reranking run on CPU.

**Step 1 — Process PDFs**

```bash
# Clean born-digital PDFs (parallel, Docling backend, no GPU needed)
./scripts/docproc-clean.sh

# Scanned/messy PDFs (sequential, DeepSeek-OCR via vLLM, needs GPU)
./scripts/docproc-messy.sh
```

**Step 2 — Download models**

```bash
uv run ras-vllm-launcher download --role all
```

**Step 3 — Embed and index**

```bash
./scripts/embed.sh
```

Starts PostgreSQL, initialises the schema, launches the Qwen3-Embedding vLLM server, indexes all
processed documents, then shuts down the embedding server.

**Step 4 — Start the chat UI**

```bash
./scripts/start-chat.sh
```

Starts PostgreSQL, the Qwen3-30B chat model via vLLM (port 8002), and the Gradio UI (port 7860).
The embedding and reranker models load on CPU on first query (~30–40s).

---

### Light stack — RTX 3090/4090 / ~16GB VRAM

Chat model: Qwen3-8B-AWQ (~5GB). Embedding (BGE-M3) and reranker (BGE-Reranker-v2-m3) on GPU
(~1.1GB each). Total ~16GB with `gpu-memory-utilization=0.45`. Uses a separate database
(`raskl_rag_light`).

**Step 1 — Process PDFs** (same as heavy stack above)

**Step 2 — Download models**

```bash
uv run ras-vllm-launcher download --role all-light
```

**Step 3 — Embed and index**

```bash
./scripts/embed-light.sh
# Skip model download if already done:
./scripts/embed-light.sh --skip-download
```

**Step 4 — Start the chat UI**

```bash
./scripts/start-chat-light.sh
```

---

## HTML debug reports

After running docproc, inspect extraction results with an interactive HTML report:

```bash
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10
```

## Serverless deployment (AWS)

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full details. The serverless stack uses:

- AWS Lambda (chat + docproc) with Bedrock for all model inference
- Neon (serverless PostgreSQL + pgvector)
- S3 for PDF upload and docproc output
- Terraform in `infra/` to provision everything
- GitHub Actions for deploy-on-push and per-PR preview environments

## Testing

```bash
# docproc end-to-end tests (uses real PDFs from docs/)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# chunker/indexer tests
uv run --package ras-chunker-indexer pytest apps/chunker_indexer/tests/ -v
```
