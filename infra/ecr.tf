# --- ECR repositories for Lambda container images ---

resource "aws_ecr_repository" "rag_api" {
  name                 = "${local.prefix}-rag-api"
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

resource "aws_ecr_lifecycle_policy" "rag_api" {
  repository = aws_ecr_repository.rag_api.name

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

import {
  to = aws_ecr_repository.admin
  id = "raskl-admin"
}

resource "aws_ecr_repository" "admin" {
  name                 = "${local.prefix}-admin"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "admin" {
  repository = aws_ecr_repository.admin.name

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

resource "aws_ecr_repository" "chunker" {
  name                 = "${local.prefix}-chunker"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "chunker" {
  repository = aws_ecr_repository.chunker.name

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
