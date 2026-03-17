#!/usr/bin/env bash
# Build ZIP packages for Alibaba Function Compute Custom Runtime deployment.
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
    local app_dirs=("$@")  # App dirs relative to ROOT (e.g. "apps/rag_engine")

    echo ""
    echo "=== Building $app_name ==="

    local workdir="$OUT/.work-$app_name"
    rm -rf "$workdir"
    mkdir -p "$workdir/code"

    # Install pip dependencies for each app
    for app_dir in "${app_dirs[@]}"; do
        echo "  pip install $ROOT/$app_dir[cloud] → $workdir/code"
        uv pip install \
            --target "$workdir/code" \
            --reinstall \
            --quiet \
            "$ROOT/$app_dir[cloud]"
    done

    # Overwrite source from app dirs (pip install puts editable packages as stubs)
    for app_dir in "${app_dirs[@]}"; do
        local src_dir="$ROOT/$app_dir/src"
        if [ -d "$src_dir" ]; then
            echo "  Copying source from $app_dir/src"
            cp -r "$src_dir"/* "$workdir/code/"
        fi
    done

    # Write bootstrap — uses Python 3.10 from the Python310 FC layer (/opt/python3.10/bin/)
    cat > "$workdir/code/bootstrap" << BOOT
#!/bin/bash
cd /code
export PATH=/opt/python3.10/bin:\$PATH
export PYTHONPATH=/code
exec python3.10 -m $entrypoint
BOOT
    chmod +x "$workdir/code/bootstrap"

    # Strip to reduce size
    find "$workdir/code" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$workdir/code" -name "*.pyc" -delete 2>/dev/null || true
    rm -rf "$workdir/code"/*.dist-info 2>/dev/null || true
    rm -rf "$workdir/code"/bin 2>/dev/null || true
    # Remove large optional packages not critical for FC runtime
    rm -rf "$workdir/code"/lingua* 2>/dev/null || true  # 292MB — language detection is non-critical
    rm -rf "$workdir/code"/rapidocr* 2>/dev/null || true
    rm -rf "$workdir/code"/onnxruntime* 2>/dev/null || true
    rm -rf "$workdir/code"/numpy/tests 2>/dev/null || true

    # Create ZIP
    (cd "$workdir/code" && zip -qr "$OUT/$app_name.zip" .)

    local size
    size=$(du -sh "$OUT/$app_name.zip" | cut -f1)
    echo "  → $app_name.zip ($size)"
    rm -rf "$workdir"
}

build_app "rag-api" "ras_rag_engine.api" \
    "apps/rag_engine"

build_app "admin" "ras_admin.api" \
    "apps/admin"

build_app "docproc" "ras_docproc.fc_handler" \
    "apps/docproc" "apps/chunker_indexer"

build_app "chunker" "ras_chunker.fc_handler" \
    "apps/chunker_indexer"

echo ""
echo "All packages:"
ls -lh "$OUT"/*.zip
