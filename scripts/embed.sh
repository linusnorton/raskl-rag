#!/usr/bin/env bash
# Embed and index all processed documents using Cohere Embed v4.
# Requires local PostgreSQL and AWS credentials with Bedrock access.
#
# Usage:
#   ./scripts/embed.sh
source "$(dirname "$0")/lib.sh"

export_aws_env

start_postgres
uv run ras-chunker init-db

step "Indexing all documents (Cohere Embed v4)"
uv run ras-chunker index-all

ok "Embedding + indexing complete"
