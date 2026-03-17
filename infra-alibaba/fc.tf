# --- Function Compute 3.0 with Python 3.10 layer ---
# Uses custom.debian10 runtime + Python310 public layer for Python 3.10.
# Our apps start HTTP servers; FC proxies to the configured port.

locals {
  python310_layer = "acs:fc:${var.region}:official:layers/Python310/versions/3"
}

# --- Code packages in OSS (hash in key forces FC to reload code) ---

resource "alicloud_oss_bucket_object" "rag_api_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/rag-api-${filemd5("${path.module}/packages/rag-api.zip")}.zip"
  source = "${path.module}/packages/rag-api.zip"
}

resource "alicloud_oss_bucket_object" "docproc_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/docproc-${filemd5("${path.module}/packages/docproc.zip")}.zip"
  source = "${path.module}/packages/docproc.zip"
}

resource "alicloud_oss_bucket_object" "chunker_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/chunker-${filemd5("${path.module}/packages/chunker.zip")}.zip"
  source = "${path.module}/packages/chunker.zip"
}

resource "alicloud_oss_bucket_object" "admin_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/admin-${filemd5("${path.module}/packages/admin.zip")}.zip"
  source = "${path.module}/packages/admin.zip"
}

# --- RAG API (HTTP, FastAPI on port 9000) ---

resource "alicloud_fcv3_function" "rag_api" {
  function_name = "${local.prefix}-rag-api"
  runtime       = "custom.debian10"
  handler       = "not-used"
  timeout       = 300
  memory_size   = 2048
  cpu           = 1
  disk_size     = 512
  role          = alicloud_ram_role.fc_exec.arn
  layers        = [local.python310_layer]

  code {
    oss_bucket_name = alicloud_oss_bucket.docs.id
    oss_object_name = alicloud_oss_bucket_object.rag_api_pkg.key
  }

  custom_runtime_config {
    command = ["bash", "/code/bootstrap"]
    args    = [""]
    port    = 9000
  }

  environment_variables = {
    CHAT_LLM_PROVIDER                 = "model_studio"
    CHAT_MODEL_STUDIO_API_KEY         = var.model_studio_api_key
    CHAT_MODEL_STUDIO_BASE_URL        = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL_STUDIO_CHAT_MODEL_ID   = var.chat_model_id
    CHAT_MODEL_STUDIO_EMBED_MODEL_ID  = var.embed_model_id
    CHAT_MODEL_STUDIO_RERANK_MODEL_ID = var.rerank_model_id
    CHAT_EMBED_DIMENSIONS             = tostring(var.embed_dimensions)
    CHAT_LLM_THINKING_BUDGET          = "2048"
    CHAT_RERANK_INSTRUCTION           = "Given a user question about historical JMBRAS and Swettenham journal documents, judge whether the document passage is relevant"
    CHAT_DATABASE_DSN                 = local.neon_dsn
    CHAT_API_PORT                     = "9000"
    CHAT_API_KEY                      = var.rag_api_key
    CHAT_WEB_SEARCH_ENABLED           = "true"
    CHAT_S3_BUCKET                    = alicloud_oss_bucket.docs.id
    PYTHONPATH                        = "/code"
  }
}

resource "alicloud_fcv3_trigger" "rag_api_http" {
  function_name = alicloud_fcv3_function.rag_api.function_name
  trigger_name  = "http-trigger"
  trigger_type  = "http"
  qualifier     = "LATEST"
  trigger_config = jsonencode({
    authType = "anonymous"
    methods  = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
  })
}

# --- DocProc (OSS-triggered, 2hr timeout) ---

resource "alicloud_fcv3_function" "docproc" {
  function_name = "${local.prefix}-docproc"
  runtime       = "custom.debian10"
  handler       = "not-used"
  timeout       = 7200
  memory_size   = 3008
  cpu           = 2
  disk_size     = 10240
  role          = alicloud_ram_role.fc_exec.arn
  layers        = [local.python310_layer]

  code {
    oss_bucket_name = alicloud_oss_bucket.docs.id
    oss_object_name = alicloud_oss_bucket_object.docproc_pkg.key
  }

  custom_runtime_config {
    command = ["bash", "/code/bootstrap"]
    args    = [""]
    port    = 9000
  }

  environment_variables = {
    DOCS_BUCKET                       = alicloud_oss_bucket.docs.id
    DOCPROC_LLM_PROVIDER              = "model_studio"
    DOCPROC_MODEL_STUDIO_API_KEY      = var.model_studio_api_key
    DOCPROC_MODEL_STUDIO_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DOCPROC_MODEL_STUDIO_OCR_MODEL_ID = var.ocr_model_id
    OSS_ACCESS_KEY_ID                 = var.access_key_id
    OSS_ACCESS_KEY_SECRET             = var.access_key_secret
    ALIBABA_REGION                    = var.region
    PYTHONPATH                        = "/code"
  }
}

# NOTE: docproc OSS trigger may conflict with old FC 2.0 trigger still on the bucket.
# Delete old triggers from FC console first, then uncomment:
# resource "alicloud_fcv3_trigger" "docproc_oss" { ... }

# --- Chunker (OSS-triggered, 1hr timeout) ---

resource "alicloud_fcv3_function" "chunker" {
  function_name = "${local.prefix}-chunker"
  runtime       = "custom.debian10"
  handler       = "not-used"
  timeout       = 3600
  memory_size   = 1024
  cpu           = 0.5
  disk_size     = 512
  role          = alicloud_ram_role.fc_exec.arn
  layers        = [local.python310_layer]

  code {
    oss_bucket_name = alicloud_oss_bucket.docs.id
    oss_object_name = alicloud_oss_bucket_object.chunker_pkg.key
  }

  custom_runtime_config {
    command = ["bash", "/code/bootstrap"]
    args    = [""]
    port    = 9000
  }

  environment_variables = {
    CHUNKER_LLM_PROVIDER                = "model_studio"
    CHUNKER_MODEL_STUDIO_API_KEY        = var.model_studio_api_key
    CHUNKER_MODEL_STUDIO_BASE_URL       = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    CHUNKER_MODEL_STUDIO_EMBED_MODEL_ID = var.embed_model_id
    CHUNKER_EMBED_DIMENSIONS            = tostring(var.embed_dimensions)
    CHUNKER_DATABASE_DSN                = local.neon_dsn
    OSS_ACCESS_KEY_ID                   = var.access_key_id
    OSS_ACCESS_KEY_SECRET               = var.access_key_secret
    ALIBABA_REGION                      = var.region
    PYTHONPATH                          = "/code"
  }
}

resource "alicloud_fcv3_trigger" "chunker_oss" {
  function_name   = alicloud_fcv3_function.chunker.function_name
  trigger_name    = "oss-documents-trigger"
  trigger_type    = "oss"
  qualifier       = "LATEST"
  source_arn      = "acs:oss:${var.region}:${local.account_id}:${alicloud_oss_bucket.docs.id}"
  invocation_role = alicloud_ram_role.fc_exec.arn
  trigger_config = jsonencode({
    events = ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject"]
    filter = { key = { prefix = "processed/", suffix = "documents.jsonl" } }
  })
}

# --- Admin (HTTP, FastAPI on port 9000) ---

resource "alicloud_fcv3_function" "admin" {
  function_name = "${local.prefix}-admin"
  runtime       = "custom.debian10"
  handler       = "not-used"
  timeout       = 120
  memory_size   = 512
  cpu           = 0.35
  disk_size     = 512
  role          = alicloud_ram_role.fc_exec.arn
  layers        = [local.python310_layer]

  code {
    oss_bucket_name = alicloud_oss_bucket.docs.id
    oss_object_name = alicloud_oss_bucket_object.admin_pkg.key
  }

  custom_runtime_config {
    command = ["bash", "/code/bootstrap"]
    args    = [""]
    port    = 9000
  }

  environment_variables = {
    ADMIN_SECRET_KEY      = var.admin_secret_key
    ADMIN_OPEN_WEBUI_URL  = "https://swetbot2.ljn.io"
    ADMIN_DATABASE_DSN    = local.neon_dsn
    ADMIN_S3_BUCKET       = alicloud_oss_bucket.docs.id
    ADMIN_PORT            = "9000"
    OSS_ACCESS_KEY_ID     = var.access_key_id
    OSS_ACCESS_KEY_SECRET = var.access_key_secret
    ALIBABA_REGION        = var.region
    PYTHONPATH            = "/code"
  }
}

resource "alicloud_fcv3_trigger" "admin_http" {
  function_name = alicloud_fcv3_function.admin.function_name
  trigger_name  = "http-trigger"
  trigger_type  = "http"
  qualifier     = "LATEST"
  trigger_config = jsonencode({
    authType = "anonymous"
    methods  = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
  })
}
