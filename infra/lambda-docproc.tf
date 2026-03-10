# --- DocProc Lambda (S3-triggered, Qwen3 VL OCR + Pipeline + Versioned S3 Output) ---

resource "aws_lambda_function" "docproc" {
  function_name = "${local.prefix}-docproc"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.docproc.repository_url}:${var.docproc_image_tag}"
  timeout                        = 900
  memory_size                    = 3008
  reserved_concurrent_executions = 5

  ephemeral_storage {
    size = 10240  # 10 GB — large PDFs (300+ pages) generate many page images
  }

  environment {
    variables = {
      # S3
      DOCS_BUCKET = aws_s3_bucket.docs.id

      # Qwen3 VL (Bedrock)
      DOCPROC_BEDROCK_REGION      = var.aws_region
      DOCPROC_BEDROCK_OCR_MODEL_ID = "qwen.qwen3-vl-235b-a22b"

      # uv cache (Lambda filesystem is read-only except /tmp)
      UV_CACHE_DIR = "/tmp/uv-cache"
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
