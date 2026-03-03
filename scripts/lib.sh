#!/usr/bin/env bash
# Shared utilities for raskl-rag scripts. Source this, don't execute it.
#   source "$(dirname "$0")/lib.sh"

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$LIB_DIR")"
cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# --- Light-stack constants ---
LIGHT_CHAT_MODEL="Qwen/Qwen3-8B-AWQ"
LIGHT_EMBED_MODEL="./models/BAAI--bge-m3"
LIGHT_RERANK_MODEL="./models/BAAI--bge-reranker-v2-m3"

# --- Output helpers ---
step() { echo -e "\n\033[1;34m=== $1 ===\033[0m"; }
ok()   { echo -e "\033[1;32m$1\033[0m"; }
fail() { echo -e "\033[1;31m$1\033[0m"; exit 1; }

# --- vLLM helpers ---

stop_vllm() {
    uv run ras-vllm-launcher down 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
}

# wait_for_vllm PORT NAME ROLE
#   Polls /v1/models until the server is ready, with PID crash detection.
wait_for_vllm() {
    local port="$1" name="$2" role="$3"
    echo "Waiting for $name on port $port ..."
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            ok "$name ready"
            return 0
        fi
        # Fast-fail if the process died (OOM / crash)
        local pid_file="data/pids/${role}.json"
        if [ -f "$pid_file" ]; then
            local pid
            pid=$(python3 -c "import json; print(json.load(open('$pid_file'))['pid'])" 2>/dev/null || echo "")
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                echo ""
                echo "Process $pid has died. Last 20 lines of log:"
                tail -20 "data/logs/${role}.log" 2>/dev/null || true
                fail "$name process crashed. Check data/logs/${role}.log"
            fi
        fi
        sleep 1
        [ $((i % 10)) -eq 0 ] && echo "  Still waiting... (${i}s)"
    done
    fail "$name not ready after 180s. Check data/logs/"
}

# --- PostgreSQL ---

start_postgres() {
    step "Starting PostgreSQL"
    docker compose up -d
    echo "Waiting for PostgreSQL..."
    for i in $(seq 1 30); do
        if docker compose exec -T postgres pg_isready -U raskl -q 2>/dev/null; then
            ok "PostgreSQL ready"
            return 0
        fi
        sleep 1
    done
    echo "[WARN] PostgreSQL not ready after 30s, continuing anyway."
}

# --- Light-stack environment ---

export_light_env() {
    export CHAT_LLM_MODEL="$LIGHT_CHAT_MODEL"
    export CHAT_LLM_CONTEXT_WINDOW=32768
    export CHAT_EMBED_MODEL="$LIGHT_EMBED_MODEL"
    export CHAT_EMBED_DIMENSIONS=1024
    export CHAT_EMBED_TASK_PREFIX=""
    export CHAT_EMBED_DEVICE="cuda"
    export CHAT_RERANK_MODEL="$LIGHT_RERANK_MODEL"
    export CHAT_RERANK_BACKEND="cross-encoder"
    export CHAT_RERANK_DEVICE="cuda"
    export CHAT_DB_NAME="raskl_rag_light"
}
