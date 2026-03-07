FROM python:3.11-slim

# Lambda Web Adapter — bundles the Lambda Runtime Interface Client
# and proxies Lambda events as HTTP requests to the web app
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files needed for dependency resolution
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY apps/rag_engine/ apps/rag_engine/
COPY apps/chat_ui/pyproject.toml apps/chat_ui/pyproject.toml
COPY apps/docproc/pyproject.toml apps/docproc/pyproject.toml
COPY apps/chunker_indexer/pyproject.toml apps/chunker_indexer/pyproject.toml

# Stub out other workspace members so uv resolves without their source
RUN mkdir -p apps/chat_ui/src/ras_chat_ui && touch apps/chat_ui/src/ras_chat_ui/__init__.py
RUN mkdir -p apps/docproc/src/ras_docproc && touch apps/docproc/src/ras_docproc/__init__.py
RUN mkdir -p apps/chunker_indexer/src/ras_chunker && touch apps/chunker_indexer/src/ras_chunker/__init__.py

# Install rag_engine with cloud dependencies
RUN uv sync --package ras-rag-engine --extra cloud --no-dev --frozen

# Lambda Web Adapter config
ENV PORT=8000
ENV AWS_LWA_INVOKE_MODE=response_stream
ENV AWS_LWA_PORT=8000
ENV AWS_LWA_READINESS_CHECK_PATH=/v1/models
ENV UV_CACHE_DIR=/tmp/uv-cache

# Bedrock provider defaults
ENV CHAT_LLM_PROVIDER=bedrock
ENV CHAT_EMBED_PROVIDER=bedrock
ENV CHAT_RERANK_PROVIDER=bedrock

# Start FastAPI RAG API — Lambda Web Adapter proxies Lambda events to port 8000
CMD ["/app/.venv/bin/python", "-m", "ras_rag_engine.api"]
