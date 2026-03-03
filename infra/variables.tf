variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
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

# --- Bedrock Model IDs (swap models without code changes) ---

variable "chat_model_id" {
  description = "Bedrock model ID for chat LLM"
  type        = string
  default     = "qwen.qwen3-32b-v1:0"
}

variable "embed_model_id" {
  description = "Bedrock model ID for embeddings"
  type        = string
  default     = "cohere.embed-multilingual-v3"
}

variable "rerank_model_id" {
  description = "Bedrock model ID for reranking"
  type        = string
  default     = "cohere.rerank-v3-5:0"
}

variable "embed_dimensions" {
  description = "Embedding vector dimensions"
  type        = number
  default     = 1024
}

# --- Container image tags (set by CI/CD) ---

variable "chat_image_tag" {
  description = "Docker image tag for chat Lambda"
  type        = string
  default     = "latest"
}

variable "docproc_image_tag" {
  description = "Docker image tag for docproc Lambda"
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
