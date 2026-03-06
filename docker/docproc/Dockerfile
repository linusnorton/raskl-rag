FROM python:3.11-slim

# Install system dependencies for docling/pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files needed for dependency resolution
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY apps/docproc/ apps/docproc/
COPY apps/chunker_indexer/ apps/chunker_indexer/
COPY apps/chat_ui/pyproject.toml apps/chat_ui/pyproject.toml
COPY apps/vllm_launcher/pyproject.toml apps/vllm_launcher/pyproject.toml

# Stub out other workspace members
RUN mkdir -p apps/chat_ui/src/ras_chat_ui && touch apps/chat_ui/src/ras_chat_ui/__init__.py
RUN mkdir -p apps/vllm_launcher/src/ras_vllm_launcher && touch apps/vllm_launcher/src/ras_vllm_launcher/__init__.py

# Install docproc + chunker with cloud dependencies
RUN uv sync --package ras-docproc --package ras-chunker-indexer --extra cloud --no-dev --frozen

# Install Lambda Runtime Interface Client
RUN uv pip install awslambdaric

# Lambda RIC entrypoint
ENTRYPOINT ["/app/.venv/bin/python", "-m", "awslambdaric"]
CMD ["ras_docproc.lambda_handler.lambda_handler"]
