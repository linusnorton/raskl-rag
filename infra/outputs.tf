output "chat_url" {
  description = "Chat UI Function URL"
  value       = aws_lambda_function_url.chat.function_url
}

output "upload_url" {
  description = "Upload page Function URL"
  value       = aws_lambda_function_url.upload.function_url
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

output "ecr_chat_repo" {
  description = "ECR repository URL for chat image"
  value       = aws_ecr_repository.chat.repository_url
}

output "ecr_docproc_repo" {
  description = "ECR repository URL for docproc image"
  value       = aws_ecr_repository.docproc.repository_url
}
