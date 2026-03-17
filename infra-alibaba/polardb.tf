# --- Neon Serverless PostgreSQL (scale-to-zero, same as AWS stack) ---
# Using Neon instead of PolarDB for true serverless scale-to-zero.
# Neon is cloud-agnostic — works from both AWS and Alibaba Cloud.

# Note: Neon resources are managed via the neon provider.
# The DSN is constructed from the neon_project outputs.

# Neon project in ap-southeast-1 (Singapore)
resource "neon_project" "alibaba" {
  name                      = "raskl-rag-alibaba"
  region_id                 = "aws-ap-southeast-1"
  history_retention_seconds = 21600
}

resource "neon_role" "app" {
  project_id = neon_project.alibaba.id
  branch_id  = neon_project.alibaba.default_branch_id
  name       = "raskl_app"
}

resource "neon_database" "main" {
  project_id = neon_project.alibaba.id
  branch_id  = neon_project.alibaba.default_branch_id
  name       = "raskl_rag"
  owner_name = neon_role.app.name
}

resource "neon_database" "open_webui" {
  project_id = neon_project.alibaba.id
  branch_id  = neon_project.alibaba.default_branch_id
  name       = "open_webui"
  owner_name = neon_role.app.name
}

locals {
  neon_host            = neon_project.alibaba.database_host
  neon_dsn             = "postgresql://${neon_role.app.name}:${neon_role.app.password}@${neon_project.alibaba.database_host}/${neon_database.main.name}?sslmode=require"
  neon_open_webui_dsn  = "postgresql://${neon_role.app.name}:${neon_role.app.password}@${neon_project.alibaba.database_host}/${neon_database.open_webui.name}?sslmode=require"
}
