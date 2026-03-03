# --- DocProc Lambda (S3-triggered, Docling + Chunk + Embed + Index) ---

resource "aws_lambda_function" "docproc" {
  function_name = "${local.prefix}-docproc"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.docproc.repository_url}:${var.docproc_image_tag}"
  timeout       = 900
  memory_size   = 8192

  environment {
    variables = {
      # S3
      DOCS_BUCKET = aws_s3_bucket.docs.id

      # Embedding provider
      CHUNKER_EMBED_PROVIDER        = "bedrock"
      CHUNKER_BEDROCK_REGION        = var.aws_region
      CHUNKER_BEDROCK_EMBED_MODEL_ID = var.embed_model_id
      CHUNKER_EMBED_DIMENSIONS      = tostring(var.embed_dimensions)
      CHUNKER_EMBED_TASK_PREFIX     = "search_document: "

      # Database (Neon)
      CHUNKER_DATABASE_DSN = local.neon_dsn
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}

# Allow S3 to invoke DocProc Lambda
resource "aws_lambda_permission" "s3_invoke_docproc" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.docproc.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.docs.arn
}
