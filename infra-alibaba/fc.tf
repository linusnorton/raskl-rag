# --- Function Compute (Custom Runtime, ZIP code packages from OSS) ---

resource "alicloud_fc_service" "main" {
  name        = "${local.prefix}-service"
  description = "raskl-rag serverless functions"
  role        = alicloud_ram_role.fc_exec.arn

  log_config {
    project  = alicloud_log_project.main.project_name
    logstore = alicloud_log_store.fc.logstore_name
  }
}

# --- Code packages in OSS ---

resource "alicloud_oss_bucket_object" "rag_api_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/rag-api.zip"
  source = "${path.module}/packages/rag-api.zip"
}

resource "alicloud_oss_bucket_object" "docproc_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/docproc.zip"
  source = "${path.module}/packages/docproc.zip"
}

resource "alicloud_oss_bucket_object" "chunker_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/chunker.zip"
  source = "${path.module}/packages/chunker.zip"
}

resource "alicloud_oss_bucket_object" "admin_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/admin.zip"
  source = "${path.module}/packages/admin.zip"
}

# --- RAG API Function (HTTP trigger, Custom Runtime) ---

resource "alicloud_fc_function" "rag_api" {
  service     = alicloud_fc_service.main.name
  name        = "${local.prefix}-rag-api"
  runtime     = "custom.debian10"
  handler     = "not-used"
  timeout     = 300
  memory_size = 2048

  code_checksum   = filemd5("${path.module}/packages/rag-api.zip")
  oss_bucket      = alicloud_oss_bucket.docs.id
  oss_key         = alicloud_oss_bucket_object.rag_api_pkg.key


  environment_variables = {
    # Provider selection
    CHAT_LLM_PROVIDER = "model_studio"

    # Model Studio configuration
    CHAT_MODEL_STUDIO_API_KEY        = var.model_studio_api_key
    CHAT_MODEL_STUDIO_BASE_URL       = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL_STUDIO_CHAT_MODEL_ID  = var.chat_model_id
    CHAT_MODEL_STUDIO_EMBED_MODEL_ID = var.embed_model_id
    CHAT_MODEL_STUDIO_RERANK_MODEL_ID = var.rerank_model_id
    CHAT_EMBED_DIMENSIONS            = tostring(var.embed_dimensions)

    # Extended thinking
    CHAT_LLM_THINKING_BUDGET = "2048"

    # Reranker domain hint
    CHAT_RERANK_INSTRUCTION = "Given a user question about historical JMBRAS and Swettenham journal documents, judge whether the document passage is relevant"

    # Database (Neon Singapore)
    CHAT_DATABASE_DSN = local.neon_dsn

    # API server
    CHAT_API_PORT = "9000"
    CHAT_API_KEY  = var.rag_api_key

    # Web search
    CHAT_WEB_SEARCH_ENABLED = "true"

    # Image serving
    CHAT_S3_BUCKET    = alicloud_oss_bucket.docs.id
    CHAT_API_BASE_URL = "https://${local.account_id}.${var.region}.fc.aliyuncs.com/2016-08-15/proxy/${alicloud_fc_service.main.name}/${local.prefix}-rag-api"

    # FC Custom Runtime
    PYTHONPATH = "/code"
  }

  lifecycle {
    ignore_changes = [code_checksum, oss_key]
  }
}

resource "alicloud_fc_trigger" "rag_api_http" {
  service  = alicloud_fc_service.main.name
  function = alicloud_fc_function.rag_api.name
  name     = "http-trigger"
  type     = "http"

  config = jsonencode({
    authType = "anonymous"
    methods  = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
  })
}

# --- DocProc Function (OSS-triggered, Custom Runtime, 2hr timeout) ---

resource "alicloud_fc_function" "docproc" {
  service     = alicloud_fc_service.main.name
  name        = "${local.prefix}-docproc"
  runtime     = "custom.debian10"
  handler     = "not-used"
  timeout     = 7200  # 2 hours
  memory_size = 3008

  code_checksum   = filemd5("${path.module}/packages/docproc.zip")
  oss_bucket      = alicloud_oss_bucket.docs.id
  oss_key         = alicloud_oss_bucket_object.docproc_pkg.key




  environment_variables = {
    DOCS_BUCKET = alicloud_oss_bucket.docs.id

    # Provider selection
    DOCPROC_LLM_PROVIDER = "model_studio"

    # Model Studio
    DOCPROC_MODEL_STUDIO_API_KEY      = var.model_studio_api_key
    DOCPROC_MODEL_STUDIO_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DOCPROC_MODEL_STUDIO_OCR_MODEL_ID = var.ocr_model_id

    # OSS access (for boto3 S3-compatible client)
    OSS_ACCESS_KEY_ID     = var.access_key_id
    OSS_ACCESS_KEY_SECRET = var.access_key_secret
    ALIBABA_REGION        = var.region

    # FC Custom Runtime
    PYTHONPATH = "/code"
  }

  lifecycle {
    ignore_changes = [code_checksum, oss_key]
  }
}

resource "alicloud_fc_trigger" "docproc_oss" {
  service  = alicloud_fc_service.main.name
  function = alicloud_fc_function.docproc.name
  name     = "oss-upload-trigger"
  type     = "oss"
  source_arn = "acs:oss:${var.region}:${local.account_id}:${alicloud_oss_bucket.docs.id}"

  config = jsonencode({
    events = ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject", "oss:ObjectCreated:CompleteMultipartUpload"]
    filter = {
      key = {
        prefix = "uploads/"
        suffix = ".pdf"
      }
    }
  })

  role = alicloud_ram_role.fc_exec.arn
}

# --- Chunker Function (OSS-triggered, Custom Runtime, 1hr timeout) ---

resource "alicloud_fc_function" "chunker" {
  service     = alicloud_fc_service.main.name
  name        = "${local.prefix}-chunker"
  runtime     = "custom.debian10"
  handler     = "not-used"
  timeout     = 3600  # 1 hour
  memory_size = 1024

  code_checksum   = filemd5("${path.module}/packages/chunker.zip")
  oss_bucket      = alicloud_oss_bucket.docs.id
  oss_key         = alicloud_oss_bucket_object.chunker_pkg.key


  environment_variables = {
    # Provider selection
    CHUNKER_LLM_PROVIDER = "model_studio"

    # Model Studio
    CHUNKER_MODEL_STUDIO_API_KEY      = var.model_studio_api_key
    CHUNKER_MODEL_STUDIO_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    CHUNKER_MODEL_STUDIO_EMBED_MODEL_ID = var.embed_model_id
    CHUNKER_EMBED_DIMENSIONS          = tostring(var.embed_dimensions)

    # Database (Neon Singapore)
    CHUNKER_DATABASE_DSN = local.neon_dsn

    # OSS access (for boto3 S3-compatible client)
    OSS_ACCESS_KEY_ID     = var.access_key_id
    OSS_ACCESS_KEY_SECRET = var.access_key_secret
    ALIBABA_REGION        = var.region

    # FC Custom Runtime
    PYTHONPATH = "/code"
  }

  lifecycle {
    ignore_changes = [code_checksum, oss_key]
  }
}

resource "alicloud_fc_trigger" "chunker_oss" {
  service  = alicloud_fc_service.main.name
  function = alicloud_fc_function.chunker.name
  name     = "oss-documents-trigger"
  type     = "oss"
  source_arn = "acs:oss:${var.region}:${local.account_id}:${alicloud_oss_bucket.docs.id}"

  config = jsonencode({
    events = ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject"]
    filter = {
      key = {
        prefix = "processed/"
        suffix = "documents.jsonl"
      }
    }
  })

  role = alicloud_ram_role.fc_exec.arn
}

# --- Admin Function (HTTP trigger, Custom Runtime) ---

resource "alicloud_fc_function" "admin" {
  service     = alicloud_fc_service.main.name
  name        = "${local.prefix}-admin"
  runtime     = "custom.debian10"
  handler     = "not-used"
  timeout     = 120
  memory_size = 512

  code_checksum   = filemd5("${path.module}/packages/admin.zip")
  oss_bucket      = alicloud_oss_bucket.docs.id
  oss_key         = alicloud_oss_bucket_object.admin_pkg.key


  environment_variables = {
    ADMIN_SECRET_KEY      = var.admin_secret_key
    ADMIN_OPEN_WEBUI_URL  = "https://swetbot2.ljn.io"
    ADMIN_DATABASE_DSN    = local.neon_dsn
    ADMIN_S3_BUCKET       = alicloud_oss_bucket.docs.id
    ADMIN_PORT            = "9000"

    # OSS access
    OSS_ACCESS_KEY_ID     = var.access_key_id
    OSS_ACCESS_KEY_SECRET = var.access_key_secret
    ALIBABA_REGION        = var.region

    # FC Custom Runtime
    PYTHONPATH = "/code"
  }

  lifecycle {
    ignore_changes = [code_checksum, oss_key]
  }
}

resource "alicloud_fc_trigger" "admin_http" {
  service  = alicloud_fc_service.main.name
  function = alicloud_fc_function.admin.name
  name     = "http-trigger"
  type     = "http"

  config = jsonencode({
    authType = "anonymous"
    methods  = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
  })
}
