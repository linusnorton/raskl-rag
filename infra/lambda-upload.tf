# --- Upload Lambda (HTML form + password + S3 presigned URL) ---

data "archive_file" "upload" {
  type        = "zip"
  source_dir  = "${path.module}/../apps/upload"
  output_path = "${path.module}/.build/upload.zip"
}

resource "aws_lambda_function" "upload" {
  function_name    = "${local.prefix}-upload"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.upload.output_path
  source_code_hash = data.archive_file.upload.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      UPLOAD_PASSWORD_HASH = var.upload_password_hash
      DOCS_BUCKET          = aws_s3_bucket.docs.id
    }
  }
}

