# --- DB Init Lambda (one-shot: CREATE EXTENSION + DDL on Neon) ---
# The zip is pre-built by CI (see .github/workflows/deploy.yml "Build db-init Lambda package" step).

resource "aws_lambda_function" "db_init" {
  function_name    = "${local.prefix}-db-init"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  filename         = "${path.module}/.build/db-init.zip"
  source_code_hash = filebase64sha256("${path.module}/lambda-db-init/handler.py")
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      DATABASE_DSN     = local.neon_dsn
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
    }
  }
}

# Invoke the DB init Lambda once after creation/update
resource "terraform_data" "db_init_invoke" {
  triggers_replace = [
    aws_lambda_function.db_init.source_code_hash,
  ]

  provisioner "local-exec" {
    command = <<-EOT
      aws lambda invoke \
        --function-name ${aws_lambda_function.db_init.function_name} \
        --region ${local.region} \
        --payload '{}' \
        /dev/stdout
    EOT
  }

  depends_on = [aws_lambda_function.db_init]
}
