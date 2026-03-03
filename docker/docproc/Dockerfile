FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for docling/pymupdf
RUN dnf install -y gcc gcc-c++ && dnf clean all

# Install Python dependencies
RUN pip install --no-cache-dir \
    docling \
    pymupdf \
    pillow \
    "lingua-language-detector>=2.0" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0" \
    click \
    rich \
    tqdm \
    jinja2 \
    httpx \
    "psycopg[binary]>=3.1" \
    "pgvector>=0.3" \
    "boto3>=1.34"

# Copy application code
COPY apps/docproc/src/ras_docproc ${LAMBDA_TASK_ROOT}/ras_docproc
COPY apps/chunker_indexer/src/ras_chunker ${LAMBDA_TASK_ROOT}/ras_chunker

# Lambda handler
CMD ["ras_docproc.lambda_handler.lambda_handler"]
