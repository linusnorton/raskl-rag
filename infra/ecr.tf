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

# Lifecycle policy: keep only the last 5 images per repo.
#
# This was previously scoped to tagStatus = "untagged", which never matched
# anything: CI tags every push with the git SHA, so no image was ever untagged
# and nothing was ever expired. By Aug 2026 that had accumulated 188 GB across
# the five repos (~$18/month), 157 GB of it from 36 pre-slimming 4.36 GB
# docproc images. Scoped to "any" so tagged images age out too.

resource "aws_ecr_lifecycle_policy" "rag_api" {
  repository = aws_ecr_repository.rag_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
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
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
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
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
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
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}
