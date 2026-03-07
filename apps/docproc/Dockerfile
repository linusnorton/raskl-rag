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
COPY apps/rag_engine/pyproject.toml apps/rag_engine/pyproject.toml

# Stub out other workspace members
RUN mkdir -p apps/rag_engine/src/ras_rag_engine && touch apps/rag_engine/src/ras_rag_engine/__init__.py

# Install docproc + chunker with cloud dependencies
RUN uv sync --package ras-docproc --package ras-chunker-indexer --extra cloud --no-dev --frozen

# Install Lambda Runtime Interface Client
RUN uv pip install awslambdaric

# Lambda RIC entrypoint
ENTRYPOINT ["/app/.venv/bin/python", "-m", "awslambdaric"]
CMD ["ras_docproc.lambda_handler.lambda_handler"]
