#!/usr/bin/env bash
# Heavy stack: start PostgreSQL + Qwen3-Embedding, index all documents into pgvector.
#
# Usage:
#   ./scripts/embed.sh                  # default settings
#   ./scripts/embed.sh --gpu-mem=0.90   # GPU memory utilization
source "$(dirname "$0")/lib.sh"

GPU_MEM=0.85

for arg in "$@"; do
    case "$arg" in
        --gpu-mem=*) GPU_MEM="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

trap 'stop_vllm' EXIT

start_postgres
uv run ras-chunker init-db

step "Starting Qwen3-Embedding server"
stop_vllm
uv run ras-vllm-launcher embed --gpu-memory-utilization "$GPU_MEM" --no-wait
wait_for_vllm 8001 "Qwen3-Embedding" embed

step "Indexing all documents into pgvector"
uv run ras-chunker index-all

step "Stopping embedding server"
# EXIT trap handles cleanup
ok "Embedding + indexing complete"
