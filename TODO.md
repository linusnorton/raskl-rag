# Deployment TODO

Checklist for getting the AWS serverless stack live.

## Accounts & Access

- [ ] **AWS account** — ensure you have one with admin access (or sufficient IAM permissions)
- [ ] **Enable Bedrock model access** — go to AWS Console → Bedrock → Model access, request access for:
  - Qwen3-32B (`qwen.qwen3-32b-v1:0`)
  - Cohere Embed Multilingual v3 (`cohere.embed-multilingual-v3`)
  - Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`)
  - Note: model access requests can take minutes to hours depending on the model
- [ ] **Neon account** — sign up at https://neon.tech (free tier is sufficient)
- [ ] **Neon API key** — create at https://console.neon.tech/app/settings/api-keys

## AWS Setup (one-time)

- [ ] **Install AWS CLI** and configure credentials (`aws configure`)
- [ ] **Install Terraform** >= 1.5 (`brew install terraform` or https://developer.hashicorp.com/terraform/install)
- [ ] **Create Terraform state bucket**:
  ```bash
  aws s3 mb s3://raskl-terraform-state --region us-east-1
  ```
- [ ] **Choose an upload password** and generate the SHA-256 hash:
  ```bash
  python3 -c "import hashlib; print(hashlib.sha256(input('Password: ').encode()).hexdigest())"
  ```
  Save the hash — you'll need it for Terraform and GitHub Secrets.

## First Terraform Deploy (bootstrapping)

The first deploy has a chicken-and-egg: Lambdas need container images, but ECR repos are created by Terraform. Do it in two steps:

- [ ] **Step 1 — Create ECR repos only** (comment out or target):
  ```bash
  cd infra
  terraform init
  terraform apply -target=aws_ecr_repository.chat -target=aws_ecr_repository.docproc \
    -var="neon_api_key=YOUR_KEY" \
    -var="upload_password_hash=YOUR_HASH"
  ```
- [ ] **Step 2 — Build and push initial images**:
  ```bash
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  REGION=us-east-1

  aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

  docker build -f docker/chat/Dockerfile -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/raskl-chat:latest .
  docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/raskl-chat:latest

  docker build -f docker/docproc/Dockerfile -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/raskl-docproc:latest .
  docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/raskl-docproc:latest
  ```
- [ ] **Step 3 — Full Terraform apply**:
  ```bash
  terraform apply \
    -var="neon_api_key=YOUR_KEY" \
    -var="upload_password_hash=YOUR_HASH"
  ```
- [ ] **Verify outputs**:
  ```bash
  terraform output chat_url      # should return a Lambda Function URL
  terraform output upload_url    # should return the upload page URL
  ```

## GitHub Secrets

Go to GitHub repo → Settings → Secrets and variables → Actions → New repository secret:

- [ ] **`AWS_DEPLOY_ROLE_ARN`** — the IAM role ARN for GitHub Actions OIDC. Get it from:
  ```bash
  terraform output github_actions_role_arn
  ```
- [ ] **`NEON_API_KEY`** — same Neon API key used above
- [ ] **`UPLOAD_PASSWORD_HASH`** — same SHA-256 hash used above

## Verify the Stack

- [ ] **Upload page** — open the upload URL in a browser, should see the HTML form
- [ ] **Upload a test PDF** — enter password, pick a clean PDF, submit
- [ ] **Check DocProc fired** — in AWS Console → Lambda → raskl-docproc → Monitor → CloudWatch Logs
- [ ] **Check Neon has data** — connect with psql or Neon console:
  ```sql
  SELECT count(*) FROM chunks;
  SELECT doc_id, title, author FROM documents;
  ```
- [ ] **Chat UI** — open the chat URL, ask a question about the uploaded document
- [ ] **PR preview** — create a test PR, check that a preview URL comment appears

## Data Migration

- [ ] **Re-embed clean PDFs** — upload to S3 (triggers DocProc with Cohere embeddings):
  ```bash
  BUCKET=$(cd infra && terraform output -raw s3_bucket)
  for pdf in docs/clean/*.pdf; do
    aws s3 cp "$pdf" "s3://$BUCKET/uploads/$(basename "$pdf")"
  done
  ```
- [ ] **Messy PDFs** — these need DeepSeek-OCR (GPU), so process locally then upload JSONL:
  ```bash
  # Process locally
  uv run ras-docproc run --pdf "docs/messy/file.pdf" --backend deepseek
  # Then re-chunk with Bedrock embeddings... (TODO: add a script for this)
  ```
  Note: messy PDFs can't go through the Lambda pipeline since Docling doesn't handle scanned text well. You'll need a local-to-cloud indexing script that chunks locally but embeds via Bedrock.

## Post-Deploy Cleanup

- [ ] **Store Terraform variables** — create `infra/terraform.tfvars` (gitignored) or use a secrets manager:
  ```hcl
  neon_api_key         = "..."
  upload_password_hash = "..."
  ```
- [ ] **Add `infra/.build/` to `.gitignore`** — Terraform creates zip artifacts there
- [ ] **Add `*.tfstate*` to `.gitignore`** — state is in S3, but local copies may appear

## Future Work

- [ ] **Messy PDF cloud pipeline** — the current Lambda pipeline only supports Docling (clean PDFs). Messy/scanned PDFs need DeepSeek-OCR which requires a GPU. Options to add:
  1. **`scripts/index-messy-cloud.sh`** — a local script that runs docproc with DeepSeek-OCR locally (GPU), then re-chunks and embeds via Bedrock (not local vLLM), and indexes directly into Neon. This avoids needing to run a local embedding server — just `CHUNKER_EMBED_PROVIDER=bedrock CHUNKER_DATABASE_DSN=<neon_dsn> uv run ras-chunker index --doc-id DOC_ID`.
  2. **Upload pre-processed JSONL to S3** — add a second S3 trigger path (e.g. `uploads-jsonl/{doc_id}/`) that skips docproc and goes straight to chunk → embed → index. The DocProc Lambda handler would detect whether it received a PDF or a JSONL directory and skip extraction accordingly.
  3. **GPU-enabled cloud processing** — use an EC2 GPU spot instance or SageMaker processing job for DeepSeek-OCR, triggered by S3 event on `uploads-messy/*.pdf`. More complex but fully automated.

  The simplest starting point is option 1: the provider abstraction already supports `CHUNKER_EMBED_PROVIDER=bedrock`, so you just need to set the Neon DSN and Bedrock config as env vars and run the existing `ras-chunker index` CLI locally.

## Known Limitations

- **Messy PDFs**: DocProc Lambda uses Docling only (no GPU for DeepSeek-OCR). Scanned/messy PDFs must be processed locally and indexed separately.
- **Neon free tier**: 0.5 GB storage, 100 compute-hours/month. If you exceed this, you'll need to upgrade (~$19/mo for the Launch plan).
- **Cold starts**: Chat Lambda ~3s, DocProc Lambda ~30-60s. DocProc is async (S3-triggered) so this is fine. Chat may feel slow on first request after idle.
- **Bedrock model availability**: Qwen3 and Cohere models must be enabled in your region. Not all models are available in all regions — us-east-1 has the best coverage.
- **psycopg2 Lambda layer**: The DB init Lambda uses a community psycopg2 layer (`arn:aws:lambda:us-east-1:898466741470:layer:psycopg2-py311:1`). If this disappears, you'll need to bundle psycopg2-binary in a zip or switch to a container.
