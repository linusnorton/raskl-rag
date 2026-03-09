variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-2"
}

variable "neon_api_key" {
  description = "Neon API key for database provisioning"
  type        = string
  sensitive   = true
}

variable "upload_password_hash" {
  description = "bcrypt hash of the upload page password"
  type        = string
  sensitive   = true
}

variable "rag_api_key" {
  description = "Bearer token for RAG API authentication"
  type        = string
  sensitive   = true
}

# --- Bedrock Model IDs (swap models without code changes) ---

variable "llm_model_id" {
  description = "Bedrock model ID for chat LLM (EU cross-region inference profile)"
  type        = string
  default     = "qwen.qwen3-235b-a22b-2507-v1:0"
}

variable "embed_model_id" {
  description = "Bedrock model ID for embeddings"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "rerank_model_id" {
  description = "Bedrock model ID for reranking"
  type        = string
  default     = "amazon.rerank-v1:0"
}

variable "rerank_region" {
  description = "AWS region for reranking (Cohere Rerank may not be in all regions)"
  type        = string
  default     = "eu-central-1"
}

variable "embed_dimensions" {
  description = "Embedding vector dimensions"
  type        = number
  default     = 1024
}

# --- Container image tags (set by CI/CD) ---

variable "rag_api_image_tag" {
  description = "Docker image tag for RAG API Lambda"
  type        = string
  default     = "latest"
}

variable "docproc_image_tag" {
  description = "Docker image tag for docproc Lambda"
  type        = string
  default     = "latest"
}

variable "chunker_image_tag" {
  description = "Docker image tag for chunker Lambda"
  type        = string
  default     = "latest"
}

# --- GitHub Actions OIDC ---

variable "github_org" {
  description = "GitHub organization or username"
  type        = string
  default     = "linusnorton"
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = "raskl-rag"
}
