#!/usr/bin/env bash
# Start RAG API (hot-reload) + Open WebUI locally, connected to Neon DB and Bedrock.
# Requires .env in repo root with CHAT_DATABASE_DSN and AWS credentials.
#
# Usage: ./scripts/jesse-mode.sh
source "$(dirname "$0")/lib.sh"

# --- Load .env ---
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    step "Loading .env"
    set -a
    source "$ENV_FILE"
    set +a
else
    fail "No .env file found at $ENV_FILE — copy .env.example and fill in values"
fi

# --- Validate required vars ---
if [[ -z "${CHAT_DATABASE_DSN:-}" ]]; then
    fail "CHAT_DATABASE_DSN is not set — add it to .env"
fi

if [[ -z "${AWS_PROFILE:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    fail "AWS credentials not set — set AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in .env"
fi

# --- Set Bedrock providers ---
# If using explicit keys, prevent export_bedrock_env from setting AWS_PROFILE
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_PROFILE=""
fi
export_bedrock_env

# --- Install dependencies ---
step "Installing dependencies"
uv sync --all-packages

# --- Cleanup on exit ---
cleanup() {
    step "Shutting down"
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null
    docker compose stop open-webui 2>/dev/null
}
trap cleanup EXIT INT TERM

# --- Start RAG API with hot reload ---
step "Starting RAG API on port 8000 (hot-reload, Neon DB, Bedrock)"
uv run uvicorn ras_rag_engine.api:app --host 0.0.0.0 --port 8000 --reload \
    --reload-dir apps/rag_engine/src &
API_PID=$!

# Wait for API to be ready
echo "Waiting for RAG API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/ >/dev/null 2>&1; then
        ok "RAG API ready (PID $API_PID)"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        fail "RAG API process died — check output above"
    fi
    sleep 1
done

# --- Start Open WebUI ---
step "Starting Open WebUI on port 3000"
docker compose up open-webui
