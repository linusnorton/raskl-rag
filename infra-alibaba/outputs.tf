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
  description = "RAG API FC3 HTTP trigger URL"
  value       = alicloud_fcv3_trigger.rag_api_http.http_trigger[0].url_internet
}

output "admin_url" {
  description = "Admin FC3 HTTP trigger URL"
  value       = alicloud_fcv3_trigger.admin_http.http_trigger[0].url_internet
}
