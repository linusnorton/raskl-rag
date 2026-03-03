# --- Chat Lambda (Gradio + Lambda Web Adapter) ---

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
      CHAT_BEDROCK_REGION         = var.aws_region
      CHAT_BEDROCK_CHAT_MODEL_ID  = var.chat_model_id
      CHAT_BEDROCK_EMBED_MODEL_ID = var.embed_model_id
      CHAT_BEDROCK_RERANK_MODEL_ID = var.rerank_model_id
      CHAT_EMBED_DIMENSIONS       = tostring(var.embed_dimensions)

      # Database (Neon)
      CHAT_DATABASE_DSN = local.neon_dsn

      # Gradio
      CHAT_GRADIO_PORT = "7860"

      # Web search enabled (no VPC = direct internet)
      CHAT_WEB_SEARCH_ENABLED = "true"

      # Lambda Web Adapter
      AWS_LWA_INVOKE_MODE = "response_stream"
      PORT                = "7860"
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}

resource "aws_lambda_function_url" "chat" {
  function_name      = aws_lambda_function.chat.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"
}
