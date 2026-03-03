# --- ECR repositories for Lambda container images ---

resource "aws_ecr_repository" "chat" {
  name                 = "${local.prefix}-chat"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_repository" "docproc" {
  name                 = "${local.prefix}-docproc"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

# Lifecycle policy: keep only last 5 untagged images
resource "aws_ecr_lifecycle_policy" "chat" {
  repository = aws_ecr_repository.chat.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 untagged images"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "docproc" {
  repository = aws_ecr_repository.docproc.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 untagged images"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}
