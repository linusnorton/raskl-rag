# --- DB Init Lambda (one-shot: CREATE EXTENSION + DDL on Neon) ---

# Build the deployment package with psycopg2-binary bundled
resource "terraform_data" "db_init_package" {
  triggers_replace = [
    filesha256("${path.module}/lambda-db-init/handler.py"),
  ]

  provisioner "local-exec" {
    command = <<-EOT
      set -e
      BUILD_DIR="${path.module}/.build/db-init-pkg"
      rm -rf "$BUILD_DIR" "${path.module}/.build/db-init.zip"
      mkdir -p "$BUILD_DIR"
      pip install --target "$BUILD_DIR" --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: psycopg2-binary -q
      cp ${path.module}/lambda-db-init/handler.py "$BUILD_DIR/"
      cd "$BUILD_DIR" && zip -r ../db-init.zip . -q
    EOT
  }
}

resource "aws_lambda_function" "db_init" {
  function_name    = "${local.prefix}-db-init"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  filename         = "${path.module}/.build/db-init.zip"
  source_code_hash = filebase64sha256("${path.module}/.build/db-init.zip")
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      DATABASE_DSN     = local.neon_dsn
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
    }
  }

  depends_on = [terraform_data.db_init_package]
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
