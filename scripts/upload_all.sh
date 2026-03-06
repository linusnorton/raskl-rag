#!/bin/bash
# Upload all PDFs to the raskl-rag upload endpoint to trigger re-processing.
# Uses presigned URLs for files > 3MB to bypass API Gateway payload limit.
# Usage: ./scripts/upload_all.sh [password]

set -euo pipefail

UPLOAD_URL="https://dehm9wnyvd.execute-api.eu-west-2.amazonaws.com/upload"
PRESIGN_URL="${UPLOAD_URL}/presign"
PASSWORD="${1:-Swettenham}"
DOCS_DIR="docs"
MAX_DIRECT_SIZE=3145728  # 3MB

uploaded=0
failed=0

for pdf in "$DOCS_DIR"/clean/*.pdf "$DOCS_DIR"/messy/*.pdf; do
    [ -f "$pdf" ] || continue
    filename=$(basename "$pdf")
    filesize=$(stat -c%s "$pdf")

    # Copy to temp file to avoid curl issues with parentheses in paths
    tmp="/tmp/upload-$$.pdf"
    cp "$pdf" "$tmp"

    echo -n "Uploading: $filename ($(numfmt --to=iec $filesize)) ... "

    if [ "$filesize" -gt "$MAX_DIRECT_SIZE" ]; then
        # Large file: get presigned URL then PUT directly to S3
        presign_resp=$(curl -s -X POST "$PRESIGN_URL" \
            -H "Content-Type: application/json" \
            -d "{\"password\": \"$PASSWORD\", \"filename\": \"$filename\"}" 2>&1)

        if echo "$presign_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'upload_url' in d else 1)" 2>/dev/null; then
            put_url=$(echo "$presign_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['upload_url'])")
            http_code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
                -H "Content-Type: application/pdf" \
                --data-binary "@${tmp}" \
                "$put_url" 2>&1)

            if [ "$http_code" = "200" ]; then
                echo "OK (presigned)"
                uploaded=$((uploaded + 1))
            else
                echo "FAILED (PUT HTTP $http_code)"
                failed=$((failed + 1))
            fi
        else
            error=$(echo "$presign_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "unknown")
            echo "FAILED (presign: $error)"
            failed=$((failed + 1))
        fi
    else
        # Small file: direct multipart upload
        response=$(curl -s -w "\n%{http_code}" \
            -F "password=$PASSWORD" \
            -F "file=@${tmp};filename=\"${filename}\"" \
            "$UPLOAD_URL" 2>&1)

        if echo "$response" | grep -q "successfully"; then
            echo "OK"
            uploaded=$((uploaded + 1))
        elif echo "$response" | grep -q "Invalid password"; then
            echo "FAILED (invalid password)"
            failed=$((failed + 1))
            rm -f "$tmp"
            echo "Aborting - password is wrong"
            exit 1
        else
            http_code=$(echo "$response" | tail -1)
            echo "FAILED (HTTP $http_code)"
            failed=$((failed + 1))
        fi
    fi

    rm -f "$tmp"
    # Small delay to avoid overwhelming the Lambda
    sleep 2
done

echo ""
echo "Done: $uploaded uploaded, $failed failed"
