# --- RAM (IAM equivalent) ---

# Service role for Function Compute
resource "alicloud_ram_role" "fc_exec" {
  role_name                      = "${local.prefix}-fc-exec"
  assume_role_policy_document    = jsonencode({
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "fc.aliyuncs.com"
          ]
        }
      }
    ]
    Version = "1"
  })
}

# OSS access policy for Function Compute
resource "alicloud_ram_policy" "fc_oss" {
  policy_name = "${local.prefix}-fc-oss"
  policy_document = jsonencode({
    Statement = [
      {
        Action = [
          "oss:GetObject",
          "oss:PutObject",
          "oss:DeleteObject",
          "oss:ListObjects",
          "oss:GetBucketInfo"
        ]
        Effect   = "Allow"
        Resource = [
          "acs:oss:*:*:${alicloud_oss_bucket.docs.id}",
          "acs:oss:*:*:${alicloud_oss_bucket.docs.id}/*"
        ]
      }
    ]
    Version = "1"
  })
}

resource "alicloud_ram_role_policy_attachment" "fc_oss" {
  policy_name = alicloud_ram_policy.fc_oss.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.fc_exec.role_name
}

# Log access policy
resource "alicloud_ram_policy" "fc_log" {
  policy_name = "${local.prefix}-fc-log"
  policy_document = jsonencode({
    Statement = [
      {
        Action = [
          "log:PostLogStoreLogs",
          "log:CreateLogStore",
          "log:GetLogStore"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
    Version = "1"
  })
}

resource "alicloud_ram_role_policy_attachment" "fc_log" {
  policy_name = alicloud_ram_policy.fc_log.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.fc_exec.role_name
}

# Function Compute invoke policy (for OSS triggers)
resource "alicloud_ram_policy" "fc_invoke" {
  policy_name = "${local.prefix}-fc-invoke"
  policy_document = jsonencode({
    Statement = [
      {
        Action = [
          "fc:InvokeFunction"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
    Version = "1"
  })
}

resource "alicloud_ram_role_policy_attachment" "fc_invoke" {
  policy_name = alicloud_ram_policy.fc_invoke.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.fc_exec.role_name
}
