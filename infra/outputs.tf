output "api_url" {
  description = "API Gateway base URL"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "open_webui_url" {
  description = "Open WebUI URL (App Runner)"
  value       = "https://${aws_apprunner_service.open_webui.service_url}"
}

output "upload_url" {
  description = "Upload page URL"
  value       = trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")
}

output "docproc_function_name" {
  description = "DocProc Lambda function name"
  value       = aws_lambda_function.docproc.function_name
}

output "s3_bucket" {
  description = "Document storage bucket"
  value       = aws_s3_bucket.docs.id
}

output "neon_host" {
  description = "Neon database host"
  value       = neon_project.main.database_host
  sensitive   = true
}

output "ecr_rag_api_repo" {
  description = "ECR repository URL for RAG API image"
  value       = aws_ecr_repository.rag_api.repository_url
}

output "ecr_docproc_repo" {
  description = "ECR repository URL for docproc image"
  value       = aws_ecr_repository.docproc.repository_url
}
