# --- Chunker Lambda (S3-triggered, Chunk + Embed + Index into Neon) ---

resource "aws_lambda_function" "chunker" {
  function_name = "${local.prefix}-chunker"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.chunker.repository_url}:${var.chunker_image_tag}"
  timeout       = 900
  memory_size   = 1024

  environment {
    variables = {
      # Embedding provider
      CHUNKER_EMBED_PROVIDER         = "bedrock"
      CHUNKER_BEDROCK_REGION         = var.aws_region
      CHUNKER_BEDROCK_EMBED_MODEL_ID = var.embed_model_id
      CHUNKER_EMBED_DIMENSIONS       = tostring(var.embed_dimensions)
      CHUNKER_EMBED_TASK_PREFIX      = ""

      # Database (Neon)
      CHUNKER_DATABASE_DSN = local.neon_dsn

      # uv cache (Lambda filesystem is read-only except /tmp)
      UV_CACHE_DIR = "/tmp/uv-cache"
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}

# Allow S3 to invoke Chunker Lambda
resource "aws_lambda_permission" "s3_invoke_chunker" {
  statement_id  = "AllowS3InvokeChunker"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chunker.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.docs.arn
}
