# --- Admin Lambda (FastAPI + Lambda Web Adapter) ---

import {
  to = aws_lambda_function.admin
  id = "raskl-admin"
}

resource "aws_lambda_function" "admin" {
  function_name = "${local.prefix}-admin"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.admin.repository_url}:${var.admin_image_tag}"
  timeout       = 120
  memory_size   = 512

  environment {
    variables = {
      # Admin config
      ADMIN_SECRET_KEY    = var.admin_secret_key
      ADMIN_OPEN_WEBUI_URL = "https://${aws_apprunner_service.open_webui.service_url}"
      ADMIN_DATABASE_DSN  = local.neon_dsn
      ADMIN_S3_BUCKET     = aws_s3_bucket.docs.id
      ADMIN_PORT          = "8000"

      # Lambda Web Adapter
      AWS_LWA_INVOKE_MODE          = "buffered"
      AWS_LWA_READINESS_CHECK_PATH = "/login"
      PORT                         = "8000"

      # uv cache
      UV_CACHE_DIR = "/tmp/uv-cache"
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}
