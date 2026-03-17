# --- ACR Personal Edition (free) ---

resource "alicloud_cr_namespace" "main" {
  name               = local.prefix
  auto_create        = false
  default_visibility = "PRIVATE"
}

resource "alicloud_cr_repo" "rag_api" {
  namespace = alicloud_cr_namespace.main.name
  name      = "rag-api"
  repo_type = "PRIVATE"
  summary   = "RAG API"
}

resource "alicloud_cr_repo" "docproc" {
  namespace = alicloud_cr_namespace.main.name
  name      = "docproc"
  repo_type = "PRIVATE"
  summary   = "DocProc"
}

resource "alicloud_cr_repo" "chunker" {
  namespace = alicloud_cr_namespace.main.name
  name      = "chunker"
  repo_type = "PRIVATE"
  summary   = "Chunker"
}

resource "alicloud_cr_repo" "admin" {
  namespace = alicloud_cr_namespace.main.name
  name      = "admin"
  repo_type = "PRIVATE"
  summary   = "Admin"
}

locals {
  ghcr_prefix = "ghcr.io/${var.github_org}"
  acr_prefix  = "registry.${var.region}.aliyuncs.com/${local.prefix}"
}
