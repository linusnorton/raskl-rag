# --- RAG API Lambda (FastAPI + Lambda Web Adapter) ---

resource "aws_lambda_function" "chat" {
  function_name = "${local.prefix}-chat"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.chat.repository_url}:${var.chat_image_tag}"
  timeout       = 300
  memory_size   = 2048

  environment {
    variables = {
      # Provider selection
      CHAT_LLM_PROVIDER    = "bedrock"
      CHAT_EMBED_PROVIDER  = "bedrock"
      CHAT_RERANK_PROVIDER = "bedrock"

      # Bedrock model configuration
      CHAT_BEDROCK_REGION          = var.aws_region
      CHAT_BEDROCK_CHAT_MODEL_ID   = var.chat_model_id
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
      CHAT_API_KEY  = var.chat_api_key

      # Web search enabled (no VPC = direct internet)
      CHAT_WEB_SEARCH_ENABLED = "true"

      # Image serving (S3 bucket for figure assets)
      CHAT_S3_BUCKET     = aws_s3_bucket.docs.id
      CHAT_API_BASE_URL  = trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")

      # Audio (S3 bucket for Transcribe temp files + custom vocabulary)
      CHAT_TRANSCRIBE_S3_BUCKET         = aws_s3_bucket.docs.id
      CHAT_TRANSCRIBE_VOCABULARY_NAME   = aws_transcribe_vocabulary.jmbras.vocabulary_name

      # Lambda Web Adapter
      AWS_LWA_INVOKE_MODE          = "buffered"
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
