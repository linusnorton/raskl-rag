# --- HTTP API Gateway for RAG API ---

resource "aws_apigatewayv2_api" "main" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true
}

# RAG API Lambda integration (default route)

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

# --- Removed: old admin routes on main API Gateway ---
# These were moved to the dedicated admin API Gateway.

removed {
  from = aws_apigatewayv2_route.admin_root
  lifecycle { destroy = true }
}

removed {
  from = aws_apigatewayv2_route.admin_get
  lifecycle { destroy = true }
}

removed {
  from = aws_apigatewayv2_route.admin_post
  lifecycle { destroy = true }
}

removed {
  from = aws_apigatewayv2_route.admin_delete
  lifecycle { destroy = true }
}

removed {
  from = aws_apigatewayv2_route.root_redirect
  lifecycle { destroy = true }
}

# --- HTTP API Gateway for Admin ---

resource "aws_apigatewayv2_api" "admin" {
  name          = "${local.prefix}-admin-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "admin" {
  api_id      = aws_apigatewayv2_api.admin.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_apigatewayv2_integration" "admin" {
  api_id                 = aws_apigatewayv2_api.admin.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.admin.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "admin_default" {
  api_id    = aws_apigatewayv2_api.admin.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.admin.id}"
}

resource "aws_lambda_permission" "apigw_admin" {
  statement_id  = "AllowAdminAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.admin.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.admin.execution_arn}/*/*"
}
