#!/usr/bin/env bash
# Start PostgreSQL (pgvector) and initialize the database schema.
source "$(dirname "$0")/lib.sh"

start_postgres
uv run ras-chunker init-db

echo ""
echo "Services ready. To start the vLLM embedding server:"
echo "  uv run ras-vllm-launcher up --embed-model ./models/Qwen--Qwen3-Embedding-8B --ports 8001"
