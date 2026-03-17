# Temporary test function to debug Python version in FC 3.0

resource "alicloud_oss_bucket_object" "test_pkg" {
  bucket = alicloud_oss_bucket.docs.id
  key    = "fc-packages/test-v2.zip"
  source = "${path.module}/packages/test.zip"
}

resource "alicloud_fcv3_function" "test" {
  function_name = "${local.prefix}-test"
  runtime       = "custom.debian10"
  handler       = "index.handler"
  timeout       = 30
  memory_size   = 256
  cpu           = 0.25
  disk_size     = 512
  layers        = [local.python310_layer]

  code {
    oss_bucket_name = alicloud_oss_bucket.docs.id
    oss_object_name = alicloud_oss_bucket_object.test_pkg.key
  }

  custom_runtime_config {
    command = ["bash", "/code/bootstrap"]
    args    = [""]
    port    = 9000
  }
}

resource "alicloud_fcv3_trigger" "test_http" {
  function_name = alicloud_fcv3_function.test.function_name
  trigger_name  = "http-trigger"
  trigger_type  = "http"
  qualifier     = "LATEST"
  trigger_config = jsonencode({
    authType = "anonymous"
    methods  = ["GET", "POST"]
  })
}

output "test_url" {
  value = alicloud_fcv3_trigger.test_http.http_trigger[0].url_internet
}
