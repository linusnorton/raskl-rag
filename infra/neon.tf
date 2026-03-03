# --- Neon Serverless PostgreSQL (free tier, scale-to-zero) ---

resource "neon_project" "main" {
  name      = "raskl-rag"
  region_id = "aws-us-east-1"

  default_endpoint_settings {
    autoscaling_limit_min_cu = 0.25
    autoscaling_limit_max_cu = 0.25
    suspend_timeout_seconds  = 300 # suspend after 5 min idle
  }
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

locals {
  neon_host     = neon_project.main.database_host
  neon_user     = neon_role.app.name
  neon_password = neon_role.app.password
  neon_dbname   = neon_database.main.name
  neon_dsn      = "postgresql://${neon_role.app.name}:${neon_role.app.password}@${neon_project.main.database_host}/${neon_database.main.name}?sslmode=require"
}
