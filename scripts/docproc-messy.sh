#!/usr/bin/env bash
# Process messy/scanned PDFs with DeepSeek-OCR via vLLM.
# Starts the OCR server, processes all PDFs, then stops the server.
#
# Usage:
#   ./scripts/docproc-messy.sh                      # default settings
#   ./scripts/docproc-messy.sh --no-force            # skip already-processed docs
#   ./scripts/docproc-messy.sh --docs-dir=path/to    # custom input directory
#   ./scripts/docproc-messy.sh --gpu-mem=0.90        # GPU memory utilization
source "$(dirname "$0")/lib.sh"

FORCE=true
DOCS_DIR="docs/messy"
GPU_MEM=0.85

for arg in "$@"; do
    case "$arg" in
        --no-force)   FORCE=false ;;
        --docs-dir=*) DOCS_DIR="${arg#*=}" ;;
        --gpu-mem=*)  GPU_MEM="${arg#*=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ ! -d "$DOCS_DIR" ]; then
    fail "Directory not found: $DOCS_DIR"
fi

MESSY_COUNT=$(find "$DOCS_DIR" -name '*.pdf' | wc -l)
if [ "$MESSY_COUNT" -eq 0 ]; then
    ok "No PDFs found in $DOCS_DIR, nothing to do."
    exit 0
fi

trap 'stop_vllm' EXIT

FORCE_FLAG=""
[ "$FORCE" = true ] && FORCE_FLAG="--force"

step "Starting DeepSeek-OCR server"
stop_vllm
uv run ras-vllm-launcher ocr --gpu-memory-utilization "$GPU_MEM" --no-wait
wait_for_vllm 8000 "DeepSeek-OCR" ocr

step "Processing $MESSY_COUNT messy PDFs (1 worker)"
echo "  docs-dir: $DOCS_DIR"
uv run ras-docproc run-all --docs-dir "$DOCS_DIR" --workers 1 $FORCE_FLAG || true

step "Stopping DeepSeek-OCR server"
# EXIT trap handles cleanup
ok "Messy PDF processing complete"
