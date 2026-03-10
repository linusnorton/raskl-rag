# --- DNS: Custom domains for swetbot.ljn.io and admin.swetbot.ljn.io ---

data "aws_route53_zone" "ljn_io" {
  name = "ljn.io."
}

# --- ACM Certificate (covers both subdomains) ---

resource "aws_acm_certificate" "swetbot" {
  domain_name               = "swetbot.ljn.io"
  subject_alternative_names = ["admin.swetbot.ljn.io"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "swetbot_cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.swetbot.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id = data.aws_route53_zone.ljn_io.zone_id
  name    = each.value.name
  type    = each.value.type
  records = [each.value.record]
  ttl     = 60

  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "swetbot" {
  certificate_arn         = aws_acm_certificate.swetbot.arn
  validation_record_fqdns = [for r in aws_route53_record.swetbot_cert_validation : r.fqdn]
}

# --- API Gateway custom domain: admin.swetbot.ljn.io ---

resource "aws_apigatewayv2_domain_name" "admin" {
  domain_name = "admin.swetbot.ljn.io"

  domain_name_configuration {
    certificate_arn = aws_acm_certificate_validation.swetbot.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }
}

resource "aws_apigatewayv2_api_mapping" "admin" {
  api_id      = aws_apigatewayv2_api.admin.id
  domain_name = aws_apigatewayv2_domain_name.admin.id
  stage       = aws_apigatewayv2_stage.admin.id
}

resource "aws_route53_record" "admin_swetbot" {
  zone_id = data.aws_route53_zone.ljn_io.zone_id
  name    = "admin.swetbot.ljn.io"
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.admin.domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.admin.domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
}

# --- App Runner custom domain: swetbot.ljn.io ---

resource "aws_apprunner_custom_domain_association" "swetbot" {
  domain_name = "swetbot.ljn.io"
  service_arn = aws_apprunner_service.open_webui.arn
}

resource "aws_route53_record" "swetbot" {
  zone_id = data.aws_route53_zone.ljn_io.zone_id
  name    = "swetbot.ljn.io"
  type    = "CNAME"
  ttl     = 300
  records = [aws_apprunner_custom_domain_association.swetbot.dns_target]
}

# App Runner requires CNAME validation records
resource "aws_route53_record" "swetbot_validation" {
  for_each = {
    for r in aws_apprunner_custom_domain_association.swetbot.certificate_validation_records : r.name => r
  }

  zone_id = data.aws_route53_zone.ljn_io.zone_id
  name    = each.value.name
  type    = each.value.type
  ttl     = 300
  records = [each.value.value]

  allow_overwrite = true
}
