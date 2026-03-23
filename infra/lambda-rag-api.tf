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

resource "aws_lambda_function_url" "rag_api" {
  function_name      = aws_lambda_function.rag_api.function_name
  authorization_type = "NONE" # Auth handled by app-level Bearer token
  invoke_mode        = "RESPONSE_STREAM"
}

resource "aws_lambda_permission" "function_url" {
  statement_id           = "AllowFunctionURLInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.rag_api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
