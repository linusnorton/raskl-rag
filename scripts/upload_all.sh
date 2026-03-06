#!/bin/bash
# Upload all PDFs to the raskl-rag upload endpoint to trigger re-processing.
# Usage: ./scripts/upload_all.sh [password]

set -euo pipefail

UPLOAD_URL="https://dehm9wnyvd.execute-api.eu-west-2.amazonaws.com/upload"
PASSWORD="${1:-Swettenham}"
DOCS_DIR="docs"

uploaded=0
failed=0
skipped=0

for pdf in "$DOCS_DIR"/clean/*.pdf "$DOCS_DIR"/messy/*.pdf; do
    [ -f "$pdf" ] || continue
    filename=$(basename "$pdf")

    # Copy to temp file to avoid curl issues with parentheses in paths
    tmp="/tmp/upload-$$.pdf"
    cp "$pdf" "$tmp"

    echo -n "Uploading: $filename ... "
    response=$(curl -s -w "\n%{http_code}" \
        -F "password=$PASSWORD" \
        -F "file=@${tmp};filename=${filename}" \
        "$UPLOAD_URL" 2>&1)

    http_code=$(echo "$response" | tail -1)
    rm -f "$tmp"

    if echo "$response" | grep -q "successfully"; then
        echo "OK"
        uploaded=$((uploaded + 1))
    elif echo "$response" | grep -q "Invalid password"; then
        echo "FAILED (invalid password)"
        failed=$((failed + 1))
        echo "Aborting - password is wrong"
        exit 1
    else
        echo "FAILED (HTTP $http_code)"
        failed=$((failed + 1))
    fi

    # Small delay to avoid overwhelming the Lambda
    sleep 2
done

echo ""
echo "Done: $uploaded uploaded, $failed failed, $skipped skipped"
