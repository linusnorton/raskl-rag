#!/usr/bin/env bash
# Process clean PDFs with Docling (CPU-only, no vLLM).
#
# Usage:
#   ./scripts/docproc-clean.sh                     # 8 workers, force re-process
#   ./scripts/docproc-clean.sh --workers=4          # 4 workers
#   ./scripts/docproc-clean.sh --no-force           # skip already-processed docs
#   ./scripts/docproc-clean.sh --docs-dir=path/to   # custom input directory
source "$(dirname "$0")/lib.sh"

WORKERS=8
FORCE=true
DOCS_DIR="docs/clean"

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

step "Processing clean PDFs with Docling ($WORKERS workers)"
echo "  docs-dir: $DOCS_DIR"
uv run ras-docproc run-all --docs-dir "$DOCS_DIR" --workers "$WORKERS" $FORCE_FLAG || true

ok "Clean PDF processing complete"
