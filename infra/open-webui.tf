# --- Open WebUI (App Runner + ECR mirror) ---

resource "aws_ecr_repository" "open_webui" {
  name                 = "${local.prefix}-open-webui"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "open_webui" {
  repository = aws_ecr_repository.open_webui.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# IAM role for App Runner to pull from ECR
resource "aws_iam_role" "apprunner_ecr" {
  name = "${local.prefix}-apprunner-ecr"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "open_webui" {
  service_name = "${local.prefix}-open-webui"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr.arn
    }

    image_repository {
      image_configuration {
        port = "8080"

        runtime_environment_variables = {
          OPENAI_API_BASE_URLS  = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/v1"
          OPENAI_API_KEYS       = var.chat_api_key
          ENABLE_OLLAMA_API     = "false"
          WEBUI_AUTH            = "true"
          AIOHTTP_CLIENT_TIMEOUT          = "300"
          ENABLE_EVALUATION_ARENA_MODELS  = "false"
          DATABASE_URL                    = local.neon_open_webui_dsn
        }
      }

      image_identifier      = "${aws_ecr_repository.open_webui.repository_url}:latest"
      image_repository_type = "ECR"
    }

    auto_deployments_enabled = true
  }

  instance_configuration {
    cpu    = "1024"   # 1 vCPU
    memory = "2048"   # 2 GB
  }

  lifecycle {
    ignore_changes = [source_configuration[0].image_repository[0].image_identifier]
  }
}
