#!/usr/bin/env bash
# Full lightweight pipeline: download models, index, launch chat.
# Thin wrapper around composable scripts.
#
# Usage:
#   ./scripts/full-pipeline-light.sh [--skip-download] [--skip-index]
source "$(dirname "$0")/lib.sh"

SKIP_DOWNLOAD=false
SKIP_INDEX=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-index)    SKIP_INDEX=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

EMBED_FLAGS=()
[ "$SKIP_DOWNLOAD" = true ] && EMBED_FLAGS+=(--skip-download)

if [ "$SKIP_INDEX" = false ]; then
    "$LIB_DIR/embed-light.sh" "${EMBED_FLAGS[@]}"
fi

exec "$LIB_DIR/start-chat-light.sh"
