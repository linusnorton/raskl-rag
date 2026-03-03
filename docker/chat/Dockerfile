FROM public.ecr.aws/lambda/python:3.11

# Install Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-web-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install Python dependencies (no torch/sentence-transformers — all inference via Bedrock)
RUN pip install --no-cache-dir \
    gradio \
    httpx \
    "psycopg[binary]>=3.1" \
    "pgvector>=0.3" \
    "pydantic-settings>=2.0" \
    "duckduckgo-search>=7.0" \
    "boto3>=1.34"

# Copy application code
COPY apps/chat_ui/src/ras_chat_ui ${LAMBDA_TASK_ROOT}/ras_chat_ui

# Lambda Web Adapter config
ENV PORT=7860
ENV AWS_LWA_INVOKE_MODE=response_stream

# Bedrock provider defaults (overridden by Lambda env vars)
ENV CHAT_LLM_PROVIDER=bedrock
ENV CHAT_EMBED_PROVIDER=bedrock
ENV CHAT_RERANK_PROVIDER=bedrock

# Launch Gradio via the app entry point
CMD ["ras_chat_ui.app.main"]
