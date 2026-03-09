# AWS Serverless Deployment

Pay-as-you-use serverless stack: Lambda for compute, Neon for pgvector (free tier, scale-to-zero), Bedrock for all model inference, S3 for document storage.

## Architecture

```
                         Internet
                            |
           +----------------+----------------+
           |                |                |
  [Upload Lambda]   [RAG API Lambda]    [PR Preview]
   Function URL     Function URL      Function URL
   (password auth)  (FastAPI + LWA)  (same image, PR config)
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
   (Qwen3 VL+Chunk| Postgres|
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

- AWS account with Bedrock model access enabled for: Qwen3-235B-A22B, Amazon Titan Embed Text v2, Amazon Rerank v1
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
├── ecr.tf               # ECR repos (raskl-rag-api, raskl-docproc)
├── lambda-rag-api.tf    # RAG API Lambda (FastAPI + Lambda Web Adapter)
├── lambda-docproc.tf    # DocProc Lambda (S3-triggered)
├── lambda-upload.tf     # Upload Lambda + Function URL
├── lambda-db-init.tf    # One-shot Lambda for DDL (CREATE EXTENSION + tables)
├── lambda-db-init/
│   └── handler.py       # Schema initialization handler
├── ../apps/upload/
│   ├── handler.py       # Upload Lambda handler
│   └── template.html    # Upload page HTML/CSS/JS
└── iam.tf               # Roles, policies, OIDC provider for GitHub Actions
```

## Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `eu-west-2` | AWS region |
| `neon_api_key` | — (sensitive) | Neon API key |
| `upload_password_hash` | — (sensitive) | SHA-256 hash of upload page password |
| `llm_model_id` | `qwen.qwen3-235b-a22b-2507-v1:0` | Bedrock chat LLM model |
| `embed_model_id` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `rerank_model_id` | `amazon.rerank-v1:0` | Bedrock rerank model |
| `rerank_region` | `eu-central-1` | AWS region for reranking (may differ from main region) |
| `embed_dimensions` | `1024` | Embedding vector dimensions |
| `rag_api_image_tag` | `latest` | RAG API Lambda container tag |
| `docproc_image_tag` | `latest` | DocProc Lambda container tag |
| `github_org` | `linusnorton` | GitHub org for OIDC |
| `github_repo` | `raskl-rag` | GitHub repo for OIDC |

## Lambda Functions

### RAG API Lambda (`raskl-rag-api`)

- **Container**: `apps/rag_engine/Dockerfile` — FastAPI + Lambda Web Adapter
- **Memory**: 2048 MB
- **Timeout**: 300s
- **Estimated image**: ~400 MB, cold start ~3s
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, etc.)
- All inference via Bedrock (LLM, embedding, reranking)

### DocProc Lambda (`raskl-docproc`)

- **Container**: `apps/docproc/Dockerfile` — Qwen3 VL (Bedrock) + PyMuPDF + chunker + Bedrock embed
- **Memory**: 8192 MB
- **Timeout**: 900s (15 min)
- **Trigger**: S3 event on `uploads/*.pdf`
- **Estimated image**: ~2-3 GB, cold start ~30-60s
- Pipeline: download PDF → docproc (Qwen3 VL via Bedrock) → versioned JSONL to S3 → diff + SES email → Chunker Lambda triggers on JSONL → chunk + embed (Bedrock Titan Embed v2) → index (Neon)
- All PDFs (clean and scanned) use the same Qwen3 VL backend.

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
| Chat LLM | Qwen3-235B-A22B (`qwen.qwen3-235b-a22b-2507-v1:0`) | MoE (235B total, 22B active). Extended thinking enabled (2048 token budget) |
| Embedding | Amazon Titan Embed Text v2 (`amazon.titan-embed-text-v2:0`) | 1024 dims, one text per call (concurrent via ThreadPoolExecutor) |
| Reranking | Amazon Rerank v1 (`amazon.rerank-v1:0`) | Via bedrock-agent-runtime Rerank API. Domain hint prepended to queries |

The embedding and reranking providers also support Cohere models (Cohere Embed Multilingual v3 and Cohere Rerank 3.5) — change the model ID in `variables.tf` to switch. Cohere Embed supports batch (96/request) and asymmetric `input_type` (search_query vs search_document).

## Provider Configuration

The codebase uses a provider abstraction to switch between local and cloud backends. In Lambda, these environment variables select Bedrock:

```bash
# RAG API Lambda (env prefix CHAT_ from RAGConfig)
CHAT_BEDROCK_REGION=eu-west-2
CHAT_BEDROCK_CHAT_MODEL_ID=qwen.qwen3-235b-a22b-2507-v1:0
CHAT_BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
CHAT_BEDROCK_RERANK_REGION=eu-central-1
CHAT_BEDROCK_RERANK_MODEL_ID=amazon.rerank-v1:0
CHAT_LLM_THINKING_BUDGET=2048
CHAT_RERANK_INSTRUCTION="Given a user question about historical JMBRAS and Swettenham journal documents, judge whether the document passage is relevant"
CHAT_DATABASE_DSN=postgresql://...@...neon.tech/raskl_rag?sslmode=require

# DocProc Lambda
CHUNKER_EMBED_PROVIDER=bedrock
CHUNKER_BEDROCK_REGION=eu-west-2
CHUNKER_BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
CHUNKER_DATABASE_DSN=postgresql://...@...neon.tech/raskl_rag?sslmode=require
```

For local development, set `AWS_PROFILE=linusnorton` and use the same Bedrock providers (see `scripts/lib.sh`).

## Initial Deployment

### 1. Create Terraform state bucket

```bash
aws s3 mb s3://raskl-terraform-state --region eu-west-2
```

### 2. Generate upload password hash

```bash
python3 -c "import hashlib; print(hashlib.sha256(input('Password: ').encode()).hexdigest())"
```

### 3. Build and push initial images

```bash
# Login to ECR (after terraform creates repos)
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.eu-west-2.amazonaws.com

# Build RAG API image
docker build -f apps/rag_engine/Dockerfile -t ACCOUNT_ID.dkr.ecr.eu-west-2.amazonaws.com/raskl-rag-api:latest .
docker push ACCOUNT_ID.dkr.ecr.eu-west-2.amazonaws.com/raskl-rag-api:latest

# Build docproc image
docker build -f apps/docproc/Dockerfile -t ACCOUNT_ID.dkr.ecr.eu-west-2.amazonaws.com/raskl-docproc:latest .
docker push ACCOUNT_ID.dkr.ecr.eu-west-2.amazonaws.com/raskl-docproc:latest
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
terraform output open_webui_url
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
2. **build-api** — Docker build RAG API → push to ECR (tagged with commit SHA)
3. **build-docproc** — Docker build → push to ECR
4. **deploy** — `terraform plan` + `apply` + update Lambda image URIs

#### `pr-preview.yml` — On PR open/sync

1. Build RAG API image tagged `pr-{N}` → push to ECR
2. Create/update Lambda `raskl-rag-api-pr-{N}` (copies prod IAM role + env vars)
3. Create Function URL (RESPONSE_STREAM)
4. Comment preview URL on the PR

PR previews share the Neon database (same indexed documents).

#### `pr-cleanup.yml` — On PR close

Deletes `raskl-rag-api-pr-{N}` Lambda + Function URL.

## Data Migration

### From local DB to Neon

The local stacks use different embedding models (Qwen3-Embedding or BGE-M3) than Bedrock
(Titan Embed v2). Vectors are not compatible across models, so chunks must be re-embedded
when migrating. Use the migration script:

```bash
# Dry run — count chunks without writing
uv run python scripts/migrate_to_neon.py --dry-run

# Full migration — reads chunks from local DB, re-embeds with Bedrock Titan, upserts to Neon
AWS_PROFILE=linusnorton uv run python scripts/migrate_to_neon.py \
  --neon-dsn "postgresql://...@...neon.tech/raskl_rag?sslmode=require"

# Single document only
AWS_PROFILE=linusnorton uv run python scripts/migrate_to_neon.py \
  --neon-dsn "postgresql://..." --doc-id swettenham-journal-1874-1876-bbfb9df1239d
```

The script is idempotent (uses upsert). It requires AWS credentials with Bedrock
`InvokeModel` permission for Titan Embed v2.

### New documents via S3 upload

Upload PDFs to S3 to trigger the DocProc Lambda (Qwen3 VL → chunk → Bedrock embed → Neon):

```bash
aws s3 cp "docs/clean/paper.pdf" "s3://$(terraform output -raw s3_bucket)/uploads/paper.pdf"
```

All PDFs (clean and scanned) are processed with Qwen3 VL via Bedrock — no separate messy/clean workflow.

## Neon Database

- Same schema as local: `documents` + `chunks` tables with `vector(1024)`, HNSW index, GIN FTS index
- Connection via standard PostgreSQL wire protocol over SSL (`sslmode=require`)
- Free tier: 0.5 GB storage, 100 compute-hours/month
- Scale-to-zero after 5 minutes idle (~1-2s resume delay on next query)
- No VPC peering needed — Neon accepts connections from public internet with SSL
