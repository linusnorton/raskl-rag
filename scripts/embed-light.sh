#!/usr/bin/env bash
# Light stack: start PostgreSQL + BGE-M3, index all documents into pgvector.
# Uses the raskl_rag_light database.
#
# Usage:
#   ./scripts/embed-light.sh                  # default settings
#   ./scripts/embed-light.sh --skip-download  # skip model download
#   ./scripts/embed-light.sh --gpu-mem=0.40   # GPU memory utilization
source "$(dirname "$0")/lib.sh"

SKIP_DOWNLOAD=false
GPU_MEM=0.30

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --gpu-mem=*)     GPU_MEM="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ "$SKIP_DOWNLOAD" = false ]; then
    step "Downloading light models"
    uv run ras-vllm-launcher download --role all-light
else
    echo "Skipping model download (--skip-download)"
fi

trap 'stop_vllm' EXIT

start_postgres

step "Initializing raskl_rag_light database"
CHUNKER_DB_NAME=raskl_rag_light uv run ras-chunker init-db

step "Starting BGE-M3 embedding server"
stop_vllm
uv run ras-vllm-launcher embed \
    --model "$LIGHT_EMBED_MODEL" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --no-wait
wait_for_vllm 8001 "BGE-M3" embed

step "Indexing all documents into raskl_rag_light"
CHUNKER_DB_NAME=raskl_rag_light \
CHUNKER_EMBED_MODEL="$LIGHT_EMBED_MODEL" \
CHUNKER_EMBED_TASK_PREFIX="" \
    uv run ras-chunker index-all

step "Stopping embedding server"
# EXIT trap handles cleanup
ok "Light embedding + indexing complete"
