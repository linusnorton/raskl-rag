#!/usr/bin/env bash
# Embed and index all processed documents using Bedrock Titan Embed v2.
# Requires local PostgreSQL and AWS credentials with Bedrock access.
#
# Usage:
#   ./scripts/embed.sh
source "$(dirname "$0")/lib.sh"

export_bedrock_env

start_postgres
uv run ras-chunker init-db

step "Indexing all documents (Bedrock Titan Embed v2)"
uv run ras-chunker index-all

ok "Embedding + indexing complete"
