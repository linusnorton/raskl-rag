# --- IAM roles and policies for Lambda functions ---

# Shared Lambda execution role
resource "aws_iam_role" "lambda_exec" {
  name = "${local.prefix}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# CloudWatch Logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Bedrock invoke access
resource "aws_iam_role_policy" "lambda_bedrock" {
  name = "${local.prefix}-bedrock"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:Converse",
          "bedrock:ConverseStream",
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*",
          "arn:aws:bedrock:*:${local.account_id}:inference-profile/eu.*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:Rerank",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "aws-marketplace:ViewSubscriptions",
          "aws-marketplace:Subscribe",
        ]
        Resource = "*"
      },
    ]
  })
}

# Transcribe access (STT via RAG API)
resource "aws_iam_role_policy" "lambda_transcribe" {
  name = "${local.prefix}-transcribe"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "transcribe:StartTranscriptionJob",
        "transcribe:GetTranscriptionJob",
        "transcribe:DeleteTranscriptionJob",
        "transcribe:CreateVocabulary",
        "transcribe:GetVocabulary",
        "transcribe:DeleteVocabulary",
      ]
      Resource = "*"
    }]
  })
}

# Polly access (TTS via RAG API)
resource "aws_iam_role_policy" "lambda_polly" {
  name = "${local.prefix}-polly"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["polly:SynthesizeSpeech"]
      Resource = "*"
    }]
  })
}

# S3 access (read/write docs bucket)
resource "aws_iam_role_policy" "lambda_s3" {
  name = "${local.prefix}-s3"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
      ]
      Resource = [
        aws_s3_bucket.docs.arn,
        "${aws_s3_bucket.docs.arn}/*",
      ]
    }]
  })
}

# --- GitHub Actions OIDC ---

data "aws_iam_openid_connect_provider" "github" {
  count = 0 # Set to 1 if OIDC provider doesn't exist yet
  url   = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "${local.prefix}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_org}/${var.github_repo}:*"
        }
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

# GitHub Actions permissions: ECR, Lambda, S3, Terraform state
resource "aws_iam_role_policy" "github_actions" {
  name = "${local.prefix}-github-actions"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "ecr:*"
        Resource = "arn:aws:ecr:${local.region}:${local.account_id}:repository/${local.prefix}-*"
      },
      {
        Effect   = "Allow"
        Action   = "lambda:*"
        Resource = "arn:aws:lambda:${local.region}:${local.account_id}:function:${local.prefix}-*"
      },
      {
        Effect = "Allow"
        Action = "s3:*"
        Resource = [
          "arn:aws:s3:::raskl-terraform-state",
          "arn:aws:s3:::raskl-terraform-state/*",
          aws_s3_bucket.docs.arn,
          "${aws_s3_bucket.docs.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = "iam:*"
        Resource = [
          "arn:aws:iam::${local.account_id}:role/${local.prefix}-*",
          "arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["execute-api:*", "apigateway:*"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["apprunner:*"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["transcribe:*"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["route53:*"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["acm:*"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["iam:CreateServiceLinkedRole"]
        Resource = "arn:aws:iam::${local.account_id}:role/aws-service-role/apprunner.amazonaws.com/*"
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = "arn:aws:iam::${local.account_id}:role/${local.prefix}-apprunner-*"
      },
    ]
  })
}

output "github_actions_role_arn" {
  description = "ARN for GitHub Actions OIDC role"
  value       = aws_iam_role.github_actions.arn
}
