# --- SLS (Simple Log Service, equivalent to CloudWatch Logs) ---

resource "alicloud_log_project" "main" {
  project_name = "${local.prefix}-logs"
  description  = "Logs for raskl-rag serverless functions"
}

resource "alicloud_log_store" "fc" {
  project_name          = alicloud_log_project.main.project_name
  logstore_name         = "function-compute"
  shard_count           = 2
  auto_split            = true
  max_split_shard_count = 8
  retention_period      = 30
}
