FROM python:3.11-slim

# Lambda Web Adapter — bundles the Lambda Runtime Interface Client
# and proxies Lambda events as HTTP requests to the web app
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files needed for dependency resolution
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY apps/admin/ apps/admin/
COPY apps/docproc/pyproject.toml apps/docproc/pyproject.toml
COPY apps/rag_engine/pyproject.toml apps/rag_engine/pyproject.toml
COPY apps/chunker_indexer/pyproject.toml apps/chunker_indexer/pyproject.toml

# Stub out other workspace members so uv resolves without their source
RUN mkdir -p apps/docproc/src/ras_docproc && touch apps/docproc/src/ras_docproc/__init__.py
RUN mkdir -p apps/rag_engine/src/ras_rag_engine && touch apps/rag_engine/src/ras_rag_engine/__init__.py
RUN mkdir -p apps/chunker_indexer/src/ras_chunker && touch apps/chunker_indexer/src/ras_chunker/__init__.py

# Install admin with cloud dependencies
RUN uv sync --package ras-admin --extra cloud --no-dev --frozen

# Lambda Web Adapter config
ENV PORT=8000
ENV AWS_LWA_INVOKE_MODE=buffered
ENV AWS_LWA_PORT=8000
ENV AWS_LWA_READINESS_CHECK_PATH=/admin/login

# uv cache (Lambda filesystem is read-only except /tmp)
ENV UV_CACHE_DIR=/tmp/uv-cache

# Start FastAPI admin app — Lambda Web Adapter proxies Lambda events to port 8000
CMD ["/app/.venv/bin/python", "-m", "ras_admin.api"]
