#!/usr/bin/env bash
# Start the RAG API server (OpenAI-compatible) using Bedrock for all model inference.
# Requires local PostgreSQL and AWS credentials with Bedrock access.
#
# Usage: ./scripts/start-api.sh
source "$(dirname "$0")/lib.sh"

export_aws_env

start_postgres

step "Starting RAG API on port 8000 (Bedrock LLM/embed/rerank)"
exec uv run ras-rag-api
