# --- RAG API Lambda (FastAPI + Lambda Web Adapter) ---

moved {
  from = aws_lambda_function.chat
  to   = aws_lambda_function.rag_api
}

resource "aws_lambda_function" "rag_api" {
  function_name = "${local.prefix}-rag-api"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.rag_api.repository_url}:${var.rag_api_image_tag}"
  timeout       = 300
  memory_size   = 2048

  environment {
    variables = {
      # Bedrock model configuration
      CHAT_BEDROCK_REGION          = var.aws_region
      CHAT_BEDROCK_CHAT_MODEL_ID   = var.llm_model_id
      CHAT_BEDROCK_EMBED_REGION    = var.embed_region
      CHAT_BEDROCK_EMBED_MODEL_ID  = var.embed_model_id
      CHAT_BEDROCK_RERANK_REGION   = var.rerank_region
      CHAT_BEDROCK_RERANK_MODEL_ID = var.rerank_model_id
      CHAT_EMBED_DIMENSIONS        = tostring(var.embed_dimensions)
      CHAT_EMBED_TASK_PREFIX       = ""

      # Extended thinking
      CHAT_LLM_THINKING_BUDGET = "2048"

      # Reranker domain hint
      CHAT_RERANK_INSTRUCTION = "Given a user question about historical JMBRAS and Swettenham journal documents, judge whether the document passage is relevant"

      # Database (Neon)
      CHAT_DATABASE_DSN = local.neon_dsn

      # API server
      CHAT_API_PORT = "8000"
      CHAT_API_KEY  = var.rag_api_key

      # Web search enabled (no VPC = direct internet)
      CHAT_WEB_SEARCH_ENABLED = "true"

      # Image serving (S3 bucket for figure assets)
      CHAT_S3_BUCKET     = aws_s3_bucket.docs.id
      CHAT_API_BASE_URL  = trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")

      # Audio (S3 bucket for Transcribe temp files + custom vocabulary)
      CHAT_TRANSCRIBE_S3_BUCKET         = aws_s3_bucket.docs.id
      CHAT_TRANSCRIBE_VOCABULARY_NAME   = aws_transcribe_vocabulary.jmbras.vocabulary_name

      # Lambda Web Adapter (response_stream for Function URL SSE streaming)
      AWS_LWA_INVOKE_MODE          = "response_stream"
      AWS_LWA_READINESS_CHECK_PATH = "/"
      AWS_LWA_INIT_BINARY          = "/opt/extensions/lambda-adapter"
      PORT                         = "8000"

      # uv cache (Lambda filesystem is read-only except /tmp)
      UV_CACHE_DIR = "/tmp/uv-cache"
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}

# --- Lambda Function URL (bypasses API Gateway 30s timeout, enables SSE streaming) ---
# Uses IAM auth + CloudFront OAC because account-level BlockPublicPolicy
# prevents NONE auth type on Lambda Function URLs.

resource "aws_lambda_function_url" "rag_api" {
  function_name      = aws_lambda_function.rag_api.function_name
  authorization_type = "AWS_IAM"
  invoke_mode        = "RESPONSE_STREAM"
}

resource "aws_lambda_permission" "cloudfront_oac" {
  statement_id  = "AllowCloudFrontOAC"
  action        = "lambda:InvokeFunctionUrl"
  function_name = aws_lambda_function.rag_api.function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.rag_api.arn
}

# --- CloudFront distribution (provides public access to IAM-auth Function URL) ---

resource "aws_cloudfront_origin_access_control" "rag_api" {
  name                              = "${local.prefix}-rag-api-oac"
  origin_access_control_origin_type = "lambda"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "rag_api" {
  comment         = "RAG API (Lambda Function URL)"
  enabled         = true
  http_version    = "http2"
  is_ipv6_enabled = true

  origin {
    domain_name              = trimsuffix(trimprefix(aws_lambda_function_url.rag_api.function_url, "https://"), "/")
    origin_id                = "rag-api-lambda"
    origin_access_control_id = aws_cloudfront_origin_access_control.rag_api.id
  }

  default_cache_behavior {
    target_origin_id       = "rag-api-lambda"
    viewer_protocol_policy = "https-only"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # No caching — pass everything through to Lambda
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" # CachingDisabled
    origin_request_policy_id = "b689b0a8-53d0-40ab-baf2-68738e2966ac" # AllViewerExceptHostHeader
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
