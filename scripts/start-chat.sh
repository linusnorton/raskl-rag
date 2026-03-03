#!/usr/bin/env bash
# Start all services needed for the RAG chat UI:
#   1. PostgreSQL (docker compose)
#   2. Chat model (vLLM on GPU, port 8002)
#   3. Chat UI (Gradio on port 7860, embeds queries on CPU)
#
# Usage: ./scripts/start-chat.sh
source "$(dirname "$0")/lib.sh"

start_postgres

echo "==> Starting chat model (Qwen3-30B-A3B) on port 8002..."
echo "    First request may be slow (model loading)."
uv run ras-vllm-launcher chat

echo "==> Starting chat UI on port 7860..."
echo "    Embedding model + reranker will load on CPU (first query takes ~30-40s)."
exec uv run ras-chat-ui
