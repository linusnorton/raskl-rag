#!/usr/bin/env bash
# Shared utilities for raskl-rag scripts. Source this, don't execute it.
#   source "$(dirname "$0")/lib.sh"

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$LIB_DIR")"
cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# --- Output helpers ---
step() { echo -e "\n\033[1;34m=== $1 ===\033[0m"; }
ok()   { echo -e "\033[1;32m$1\033[0m"; }
fail() { echo -e "\033[1;31m$1\033[0m"; exit 1; }

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

# --- Bedrock environment ---

export_bedrock_env() {
    export AWS_PROFILE="${AWS_PROFILE:-linusnorton}"
    export CHAT_LLM_PROVIDER=bedrock
    export CHAT_EMBED_PROVIDER=bedrock
    export CHAT_RERANK_PROVIDER=bedrock
    export CHAT_EMBED_TASK_PREFIX=""
    export CHAT_LLM_THINKING_BUDGET=2048
    export CHUNKER_EMBED_PROVIDER=bedrock
}
