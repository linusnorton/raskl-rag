#!/usr/bin/env bash
# Start the RAG chat UI using Bedrock for all model inference.
# Requires local PostgreSQL and AWS credentials with Bedrock access.
#
# Usage: ./scripts/start-chat.sh
source "$(dirname "$0")/lib.sh"

export_bedrock_env

start_postgres

step "Starting chat UI on port 7860 (Bedrock LLM/embed/rerank)"
exec uv run ras-chat-ui
