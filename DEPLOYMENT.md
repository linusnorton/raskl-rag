# AWS Serverless Deployment

Pay-as-you-use serverless stack: Lambda for compute, Neon for pgvector (free tier, scale-to-zero), Bedrock for all model inference, S3 for document storage.

## Architecture

```
                         Internet
                            |
           +----------------+----------------+
           |                |                |
  [Upload Lambda]   [Chat Lambda]    [PR Preview Lambda]
   Function URL     Function URL      Function URL
   (password auth)  (Gradio + LWA)   (same image, PR config)
           |           |    |                |
           v           |    v                |
     +---------+       |  +----------+      |
     |   S3    |       |  | Bedrock  |<-----+
     | Bucket  |       |  | Chat/Emb |
     +----+----+       |  | /Rerank  |
          |            |  +----------+
    S3 Event           |
          |            v
          v       +---------+
   [DocProc Lambda]|  Neon   |
   (Docling+Chunk  | Postgres|
    +Embed+Index)->|pgvector |
                   | (-> 0)  |
                   +---------+
```

No VPC required. All Lambdas run outside VPC with direct internet access to Neon (SSL) and Bedrock.

## Cost Estimates

### Idle

| Resource | Cost/mo |
|----------|---------|
| Neon free tier (0.5 GB, 100 compute-hrs) | $0.00 |
| S3 storage (~10 GB PDFs + JSONL) | ~$0.25 |
| ECR storage (~3 GB images) | ~$1.00 |
| Lambda (idle) | $0.00 |
| **Total idle** | **~$1.25/mo** |

### Per-use

| Action | Est. Cost |
|--------|-----------|
| One chat query | ~$0.005-0.02 |
| Process + index a 30-page PDF | ~$0.05-0.15 |

## Prerequisites

- AWS account with Bedrock model access enabled for: Qwen3-32B, Cohere Embed Multilingual v3, Cohere Rerank 3.5
- [Neon](https://neon.tech) account + API key (free tier works)
- Terraform >= 1.5
- Docker (for building Lambda container images)
- GitHub repository (for CI/CD)

## Terraform Structure

```
infra/
├── main.tf              # AWS + Neon providers, S3 backend
├── variables.tf         # Region, model IDs, Neon API key, upload password
├── outputs.tf           # Chat URL, Upload URL
├── neon.tf              # Neon project, database, role, connection string
├── s3.tf                # Document bucket + S3 event notification
├── ecr.tf               # ECR repos (raskl-chat, raskl-docproc)
├── lambda-chat.tf       # Chat Lambda + Function URL (RESPONSE_STREAM)
├── lambda-docproc.tf    # DocProc Lambda (S3-triggered)
├── lambda-upload.tf     # Upload Lambda + Function URL
├── lambda-db-init.tf    # One-shot Lambda for DDL (CREATE EXTENSION + tables)
├── lambda-db-init/
│   └── handler.py       # Schema initialization handler
├── lambda-upload/
│   └── handler.py       # Upload form + password auth handler
└── iam.tf               # Roles, policies, OIDC provider for GitHub Actions
```

## Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `us-east-1` | AWS region |
| `neon_api_key` | — (sensitive) | Neon API key |
| `upload_password_hash` | — (sensitive) | SHA-256 hash of upload page password |
| `chat_model_id` | `qwen.qwen3-32b-v1:0` | Bedrock chat model |
| `embed_model_id` | `cohere.embed-multilingual-v3` | Bedrock embedding model |
| `rerank_model_id` | `cohere.rerank-v3-5:0` | Bedrock rerank model |
| `embed_dimensions` | `1024` | Embedding vector dimensions |
| `chat_image_tag` | `latest` | Chat Lambda container tag |
| `docproc_image_tag` | `latest` | DocProc Lambda container tag |
| `github_org` | `linusnorton` | GitHub org for OIDC |
| `github_repo` | `raskl-rag` | GitHub repo for OIDC |

## Lambda Functions

### Chat Lambda (`raskl-chat`)

- **Container**: `docker/chat/Dockerfile` — Gradio + Lambda Web Adapter, no torch/transformers
- **Memory**: 2048 MB
- **Timeout**: 300s
- **Function URL**: RESPONSE_STREAM mode (for Gradio SSE)
- **Estimated image**: ~400 MB, cold start ~3s
- All inference via Bedrock (LLM, embedding, reranking)

### DocProc Lambda (`raskl-docproc`)

- **Container**: `docker/docproc/Dockerfile` — Docling + PyMuPDF + chunker + Bedrock embed
- **Memory**: 8192 MB
- **Timeout**: 900s (15 min)
- **Trigger**: S3 event on `uploads/*.pdf`
- **Estimated image**: ~2-3 GB, cold start ~30-60s
- Pipeline: download PDF → docproc (Docling) → chunk → embed (Bedrock Cohere) → index (Neon) → upload JSONL to S3
- Docling only — no GPU for DeepSeek-OCR in Lambda. Messy PDFs must be processed locally.

### Upload Lambda (`raskl-upload`)

- **Runtime**: Python 3.11 (zip package, no container)
- **Memory**: 256 MB
- **Timeout**: 30s
- **Function URL**: HTML form with password field + file picker
- Password validated against SHA-256 hash in environment variable
- Uploads directly to S3 `uploads/` prefix, which triggers DocProc

### DB Init Lambda (`raskl-db-init`)

- **Runtime**: Python 3.11 with psycopg2 layer
- **Invoked once** via `terraform_data` provisioner after creation
- Runs `CREATE EXTENSION IF NOT EXISTS vector` + full DDL on Neon

## Bedrock Models

| Purpose | Model | Notes |
|---------|-------|-------|
| Chat LLM | Qwen3-32B (`qwen.qwen3-32b-v1:0`) | Configurable to Qwen3-235B via Terraform variable |
| Embedding | Cohere Embed Multilingual v3 | 1024 dims, multilingual (en/ms/zh/ar), batch up to 96 texts |
| Reranking | Cohere Rerank 3.5 | Via bedrock-agent-runtime Rerank API |

### Cohere Embed input types

- Chat queries: `input_type: "search_query"`
- Document indexing: `input_type: "search_document"`

## Provider Configuration

The codebase uses a provider abstraction to switch between local and cloud backends. In Lambda, these environment variables select Bedrock:

```bash
# Chat Lambda
CHAT_LLM_PROVIDER=bedrock
CHAT_EMBED_PROVIDER=bedrock
CHAT_RERANK_PROVIDER=bedrock
CHAT_BEDROCK_REGION=us-east-1
CHAT_BEDROCK_CHAT_MODEL_ID=qwen.qwen3-32b-v1:0
CHAT_BEDROCK_EMBED_MODEL_ID=cohere.embed-multilingual-v3
CHAT_BEDROCK_RERANK_MODEL_ID=cohere.rerank-v3-5:0
CHAT_DATABASE_DSN=postgresql://...@...neon.tech/raskl_rag?sslmode=require

# DocProc Lambda
CHUNKER_EMBED_PROVIDER=bedrock
CHUNKER_BEDROCK_REGION=us-east-1
CHUNKER_BEDROCK_EMBED_MODEL_ID=cohere.embed-multilingual-v3
CHUNKER_DATABASE_DSN=postgresql://...@...neon.tech/raskl_rag?sslmode=require
```

For local development, defaults remain unchanged (`vllm`, `sentence-transformers`, `qwen3`).

## Initial Deployment

### 1. Create Terraform state bucket

```bash
aws s3 mb s3://raskl-terraform-state --region us-east-1
```

### 2. Generate upload password hash

```bash
python3 -c "import hashlib; print(hashlib.sha256(input('Password: ').encode()).hexdigest())"
```

### 3. Build and push initial images

```bash
# Login to ECR (after terraform creates repos)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build chat image
docker build -f docker/chat/Dockerfile -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/raskl-chat:latest .
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/raskl-chat:latest

# Build docproc image
docker build -f docker/docproc/Dockerfile -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/raskl-docproc:latest .
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/raskl-docproc:latest
```

### 4. Deploy with Terraform

```bash
cd infra
terraform init
terraform plan \
  -var="neon_api_key=YOUR_NEON_API_KEY" \
  -var="upload_password_hash=YOUR_SHA256_HASH"
terraform apply
```

### 5. Verify

```bash
# Get URLs
terraform output chat_url
terraform output upload_url

# Test upload
curl $(terraform output -raw upload_url)

# Check Neon database
psql "$(terraform output -raw neon_host)" -c "SELECT count(*) FROM chunks;"
```

## CI/CD — GitHub Actions

### Required Secrets

| Secret | Description |
|--------|-------------|
| `AWS_DEPLOY_ROLE_ARN` | ARN of `raskl-github-actions` IAM role (output from Terraform) |
| `NEON_API_KEY` | Neon API key |
| `UPLOAD_PASSWORD_HASH` | SHA-256 hash of upload password |

### Workflows

#### `deploy.yml` — On push to master

1. **test** — `uv sync` + pytest (chunker tests)
2. **build-chat** — Docker build → push to ECR (tagged with commit SHA)
3. **build-docproc** — Docker build → push to ECR
4. **deploy** — `terraform plan` + `apply` + update Lambda image URIs

#### `pr-preview.yml` — On PR open/sync

1. Build chat image tagged `pr-{N}` → push to ECR
2. Create/update Lambda `raskl-chat-pr-{N}` (copies prod IAM role + env vars)
3. Create Function URL (RESPONSE_STREAM)
4. Comment preview URL on the PR

PR previews share the Neon database (same indexed documents).

#### `pr-cleanup.yml` — On PR close

Deletes `raskl-chat-pr-{N}` Lambda + Function URL.

## Data Migration

To re-embed all documents with Cohere Embed Multilingual v3 (required since the local stack uses Qwen3 embeddings which are incompatible):

```bash
# Upload existing PDFs to S3 — each triggers DocProc Lambda automatically
for pdf in docs/clean/*.pdf; do
  aws s3 cp "$pdf" "s3://$(terraform output -raw s3_bucket)/uploads/$(basename "$pdf")"
done
```

Alternatively, process locally and upload JSONL directly (for messy PDFs that need DeepSeek-OCR):

```bash
# Process locally with DeepSeek-OCR
uv run ras-docproc run --pdf "docs/messy/file.pdf" --backend deepseek

# Upload processed output to S3
aws s3 sync data/out/DOC_ID/ "s3://$(terraform output -raw s3_bucket)/processed/DOC_ID/"
```

## Neon Database

- Same schema as local: `documents` + `chunks` tables with `vector(1024)`, HNSW index, GIN FTS index
- Connection via standard PostgreSQL wire protocol over SSL (`sslmode=require`)
- Free tier: 0.5 GB storage, 100 compute-hours/month
- Scale-to-zero after 5 minutes idle (~1-2s resume delay on next query)
- No VPC peering needed — Neon accepts connections from public internet with SSL
