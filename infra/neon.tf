# --- Neon Serverless PostgreSQL (free tier, scale-to-zero) ---

resource "neon_project" "main" {
  name                          = "raskl-rag"
  region_id                     = "aws-eu-west-2"
  history_retention_seconds     = 21600
}

resource "neon_database" "main" {
  project_id = neon_project.main.id
  branch_id  = neon_project.main.default_branch_id
  name       = "raskl_rag"
  owner_name = neon_role.app.name
}

resource "neon_role" "app" {
  project_id = neon_project.main.id
  branch_id  = neon_project.main.default_branch_id
  name       = "raskl_app"
}

resource "neon_database" "open_webui" {
  project_id = neon_project.main.id
  branch_id  = neon_project.main.default_branch_id
  name       = "open_webui"
  owner_name = neon_role.app.name
}

locals {
  neon_host     = neon_project.main.database_host
  neon_user     = neon_role.app.name
  neon_password = neon_role.app.password
  neon_dbname   = neon_database.main.name
  neon_dsn      = "postgresql://${neon_role.app.name}:${neon_role.app.password}@${neon_project.main.database_host}/${neon_database.main.name}?sslmode=require"
  neon_open_webui_dsn = "postgresql://${neon_role.app.name}:${neon_role.app.password}@${neon_project.main.database_host}/${neon_database.open_webui.name}?sslmode=require"
}
