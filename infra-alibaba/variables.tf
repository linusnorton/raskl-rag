variable "region" {
  description = "Alibaba Cloud region"
  type        = string
  default     = "ap-southeast-1"
}

variable "access_key_id" {
  description = "Alibaba Cloud Access Key ID"
  type        = string
  sensitive   = true
}

variable "access_key_secret" {
  description = "Alibaba Cloud Access Key Secret"
  type        = string
  sensitive   = true
}

variable "model_studio_api_key" {
  description = "Alibaba Model Studio API key"
  type        = string
  sensitive   = true
}

variable "rag_api_key" {
  description = "Bearer token for RAG API authentication"
  type        = string
  sensitive   = true
}

variable "upload_password_hash" {
  description = "bcrypt hash of the upload page password"
  type        = string
  sensitive   = true
}

variable "admin_secret_key" {
  description = "JWT signing key for admin sessions"
  type        = string
  sensitive   = true
  default     = "change-me-in-production"
}

# --- Model Studio Model IDs ---

variable "chat_model_id" {
  description = "Model Studio model ID for chat LLM"
  type        = string
  default     = "qwen3.5-397b-a17b"
}

variable "ocr_model_id" {
  description = "Model Studio model ID for OCR/vision"
  type        = string
  default     = "qwen3.5-397b-a17b"
}

variable "embed_model_id" {
  description = "Model Studio model ID for embeddings"
  type        = string
  default     = "text-embedding-v4"
}

variable "rerank_model_id" {
  description = "Model Studio model ID for reranking"
  type        = string
  default     = "gte-rerank"
}

variable "embed_dimensions" {
  description = "Embedding vector dimensions"
  type        = number
  default     = 1024
}

# --- Container image tags ---

variable "rag_api_image_tag" {
  description = "Docker image tag for RAG API"
  type        = string
  default     = "latest"
}

variable "docproc_image_tag" {
  description = "Docker image tag for docproc"
  type        = string
  default     = "latest"
}

variable "chunker_image_tag" {
  description = "Docker image tag for chunker"
  type        = string
  default     = "latest"
}

variable "admin_image_tag" {
  description = "Docker image tag for admin"
  type        = string
  default     = "latest"
}

# --- Database (Neon) ---

variable "neon_api_key" {
  description = "Neon API key for database provisioning"
  type        = string
  sensitive   = true
}

# --- GitHub Actions ---

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
