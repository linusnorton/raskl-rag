#!/usr/bin/env bash
# Full pipeline: OCR messy docs → docproc all → embed → index into pgvector
# Thin wrapper around composable scripts.
#
# Usage:
#   ./scripts/full-pipeline.sh              # run everything
#   ./scripts/full-pipeline.sh --skip-ocr   # skip DeepSeek OCR stage
#   ./scripts/full-pipeline.sh --skip-docproc  # skip docproc, only embed+index
#   ./scripts/full-pipeline.sh --workers=4  # parallel docling workers
source "$(dirname "$0")/lib.sh"

SKIP_OCR=false
SKIP_DOCPROC=false
WORKERS=8

for arg in "$@"; do
    case "$arg" in
        --skip-ocr)     SKIP_OCR=true ;;
        --skip-docproc) SKIP_DOCPROC=true ;;
        --workers=*)    WORKERS="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

step "Installing packages"
uv sync --all-packages

if [ "$SKIP_OCR" = false ] && [ "$SKIP_DOCPROC" = false ] && [ -d docs/messy ]; then
    "$LIB_DIR/docproc-messy.sh"
fi

if [ "$SKIP_DOCPROC" = false ] && [ -d docs/clean ]; then
    "$LIB_DIR/docproc-clean.sh" --workers="$WORKERS"
fi

"$LIB_DIR/embed.sh"

ok "Full pipeline complete!"
