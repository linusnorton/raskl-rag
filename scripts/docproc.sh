#!/usr/bin/env bash
# Process all PDFs (clean + messy) using Qwen3 VL via Bedrock.
#
# Usage:
#   ./scripts/docproc.sh                      # all PDFs in docs/, 8 workers
#   ./scripts/docproc.sh --workers=4          # 4 workers
#   ./scripts/docproc.sh --no-force           # skip already-processed docs
#   ./scripts/docproc.sh --docs-dir=path/to   # custom input directory
source "$(dirname "$0")/lib.sh"

export_aws_env

WORKERS=8
FORCE=true
DOCS_DIR="docs"

for arg in "$@"; do
    case "$arg" in
        --workers=*)  WORKERS="${arg#*=}" ;;
        --no-force)   FORCE=false ;;
        --docs-dir=*) DOCS_DIR="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ ! -d "$DOCS_DIR" ]; then
    fail "Directory not found: $DOCS_DIR"
fi

FORCE_FLAG=""
[ "$FORCE" = true ] && FORCE_FLAG="--force"

step "Processing all PDFs with Qwen3 VL (Bedrock, $WORKERS workers)"
echo "  docs-dir: $DOCS_DIR"
uv run ras-docproc run-all --docs-dir "$DOCS_DIR" --workers "$WORKERS" $FORCE_FLAG || true

ok "PDF processing complete"
