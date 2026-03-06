# --- S3 bucket for PDF uploads and processed output ---

resource "aws_s3_bucket" "docs" {
  bucket = "${local.prefix}-docs-${local.account_id}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket = aws_s3_bucket.docs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 event notification → DocProc Lambda (on uploads/*.pdf)
resource "aws_s3_bucket_notification" "docproc_trigger" {
  bucket = aws_s3_bucket.docs.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.docproc.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
    filter_suffix       = ".pdf"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.chunker.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "processed/"
    filter_suffix       = "documents.jsonl"
  }

  depends_on = [aws_lambda_permission.s3_invoke_docproc, aws_lambda_permission.s3_invoke_chunker]
}
