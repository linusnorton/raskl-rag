output "oss_bucket" {
  description = "Document storage bucket"
  value       = alicloud_oss_bucket.docs.id
}

output "neon_host" {
  description = "Neon database host"
  value       = neon_project.alibaba.database_host
  sensitive   = true
}

output "rag_api_url" {
  description = "RAG API Function Compute HTTP trigger URL"
  value       = "https://${local.account_id}.${var.region}.fc.aliyuncs.com/2016-08-15/proxy/${alicloud_fc_service.main.name}/${alicloud_fc_function.rag_api.name}/"
}

output "admin_url" {
  description = "Admin UI Function Compute HTTP trigger URL"
  value       = "https://${local.account_id}.${var.region}.fc.aliyuncs.com/2016-08-15/proxy/${alicloud_fc_service.main.name}/${alicloud_fc_function.admin.name}/"
}

output "fc_service" {
  description = "FC service name"
  value       = alicloud_fc_service.main.name
}
