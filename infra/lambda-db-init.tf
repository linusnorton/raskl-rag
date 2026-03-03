# --- DB Init Lambda (one-shot: CREATE EXTENSION + DDL on Neon) ---

data "archive_file" "db_init" {
  type        = "zip"
  source_dir  = "${path.module}/lambda-db-init"
  output_path = "${path.module}/.build/db-init.zip"
}

resource "aws_lambda_function" "db_init" {
  function_name    = "${local.prefix}-db-init"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.db_init.output_path
  source_code_hash = data.archive_file.db_init.output_base64sha256
  timeout          = 60
  memory_size      = 256

  layers = [
    "arn:aws:lambda:${local.region}:898466741470:layer:psycopg2-py311:1",
  ]

  environment {
    variables = {
      DATABASE_DSN     = local.neon_dsn
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
    }
  }
}

# Invoke the DB init Lambda once after creation
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
