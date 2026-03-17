#!/usr/bin/env bash
# Build ZIP packages for FC Custom Runtime targeting Python 3.10.
# Uses Docker python:3.10-slim for correct C extension ABI.
# Output: infra-alibaba/packages/{app}.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/infra-alibaba/packages"
rm -rf "$OUT"
mkdir -p "$OUT"

build_app() {
    local app_name="$1"
    local entrypoint="$2"
    shift 2
    local app_dirs=("$@")

    echo ""
    echo "=== Building $app_name ==="

    # Step 1: Export requirements using uv (fast, local)
    local export_args=()
    for app_dir in "${app_dirs[@]}"; do
        case "$app_dir" in
            apps/rag_engine) export_args+=(--package ras-rag-engine) ;;
            apps/admin) export_args+=(--package ras-admin) ;;
            apps/docproc) export_args+=(--package ras-docproc) ;;
            apps/chunker_indexer) export_args+=(--package ras-chunker-indexer) ;;
        esac
    done

    local reqfile="$OUT/${app_name}-requirements.txt"
    cd "$ROOT"
    uv export "${export_args[@]}" --extra cloud --no-dev --frozen --no-hashes 2>/dev/null \
        | grep -v '^-e ' | grep -v '^#' | grep -v '^\s*$' > "$reqfile"
    echo "  $(wc -l < "$reqfile") deps exported"

    # Step 2: Build copy commands for source dirs
    local copy_cmds=""
    for app_dir in "${app_dirs[@]}"; do
        copy_cmds="$copy_cmds cp -r /app/$app_dir/src/* /build/ &&"
    done

    # Step 3: Run everything in Docker (pip install + copy source + strip + zip)
    docker run --rm \
        -v "$ROOT:/app:ro" \
        -v "$OUT:/output" \
        -v "$reqfile:/requirements.txt:ro" \
        python:3.10-slim \
        bash -c "
            set -e
            apt-get update -qq && apt-get install -y -qq zip >/dev/null 2>&1
            pip install --quiet --no-cache-dir --target /build -r /requirements.txt 2>&1 | tail -3
            $copy_cmds true
            cat > /build/bootstrap << 'BOOT'
#!/bin/bash
cd /code
export PYTHONPATH=/code
exec /opt/python3.10/bin/python3.10 -m $entrypoint
BOOT
            chmod +x /build/bootstrap
            find /build -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            find /build -name '*.pyc' -delete 2>/dev/null || true
            rm -rf /build/*.dist-info /build/bin 2>/dev/null || true
            rm -rf /build/lingua* 2>/dev/null || true
            cd /build && zip -qr /output/$app_name.zip .
        "

    rm -f "$reqfile"
    local size
    size=$(du -sh "$OUT/$app_name.zip" | cut -f1)
    echo "  → $app_name.zip ($size)"
}

build_app "rag-api" "ras_rag_engine.api" "apps/rag_engine"
build_app "admin" "ras_admin.api" "apps/admin"
build_app "docproc" "ras_docproc.fc_handler" "apps/docproc" "apps/chunker_indexer"
build_app "chunker" "ras_chunker.fc_handler" "apps/chunker_indexer"

echo ""
echo "All packages:"
ls -lh "$OUT"/*.zip
