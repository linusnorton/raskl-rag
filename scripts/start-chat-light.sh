#!/usr/bin/env bash
# Lightweight RAG chat stack — all models in VRAM (~8GB total).
#   1. PostgreSQL (docker compose, database: raskl_rag_light)
#   2. Chat model (Qwen3-8B-AWQ on GPU, port 8002)
#   3. Chat UI (Gradio on 7860, BGE-M3 + BGE-Reranker on GPU)
source "$(dirname "$0")/lib.sh"

start_postgres

echo "==> Starting chat model (Qwen3-8B-AWQ) on port 8002..."
uv run ras-vllm-launcher chat \
  --model "$LIGHT_CHAT_MODEL" \
  --gpu-memory-utilization 0.45 \
  --max-model-len 32768

echo "==> Starting chat UI on port 7860..."
export_light_env
exec uv run ras-chat-ui
