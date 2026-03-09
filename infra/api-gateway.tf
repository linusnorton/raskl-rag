# --- HTTP API Gateway (routes to RAG API + upload Lambdas) ---

resource "aws_apigatewayv2_api" "main" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true
}

# --- Upload Lambda integration ---

resource "aws_apigatewayv2_integration" "upload" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.upload.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "upload_get" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_apigatewayv2_route" "upload_post" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /upload"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_apigatewayv2_route" "upload_presign" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /upload/presign"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_apigatewayv2_route" "upload_status" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /upload/status"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_apigatewayv2_route" "upload_reprocess" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /upload/reprocess"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_lambda_permission" "apigw_upload" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.upload.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# --- RAG API Lambda integration ---

moved {
  from = aws_apigatewayv2_integration.chat
  to   = aws_apigatewayv2_integration.rag_api
}

resource "aws_apigatewayv2_integration" "rag_api" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.rag_api.invoke_arn
  payload_format_version = "2.0"
}

moved {
  from = aws_apigatewayv2_route.chat
  to   = aws_apigatewayv2_route.rag_api
}

resource "aws_apigatewayv2_route" "rag_api" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.rag_api.id}"
}

moved {
  from = aws_lambda_permission.apigw_chat
  to   = aws_lambda_permission.apigw_rag_api
}

resource "aws_lambda_permission" "apigw_rag_api" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rag_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}
