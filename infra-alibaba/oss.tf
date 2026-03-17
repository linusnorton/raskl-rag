# --- OSS Bucket (document storage, equivalent to S3) ---

resource "alicloud_oss_bucket" "docs" {
  bucket        = "${local.prefix}-docs-${local.account_id}"
  storage_class = "Standard"
  # ACL managed separately via alicloud_oss_bucket_acl

  server_side_encryption_rule {
    sse_algorithm = "AES256"
  }

  cors_rule {
    allowed_methods = ["PUT"]
    allowed_origins = ["*"]
    allowed_headers = ["*"]
    max_age_seconds = 3600
  }
}
