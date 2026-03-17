# --- Open WebUI on ECS (tiny instance running Docker) ---

# VPC + VSwitch for ECS
resource "alicloud_vpc" "default" {
  vpc_name   = "${local.prefix}-vpc"
  cidr_block = "172.16.0.0/16"
}

resource "alicloud_vswitch" "default" {
  vpc_id     = alicloud_vpc.default.id
  cidr_block = "172.16.0.0/24"
  zone_id    = "ap-southeast-1c"
  vswitch_name = "${local.prefix}-vswitch"
}

# Security group: allow HTTP 8080 + SSH
resource "alicloud_security_group" "webui" {
  name   = "${local.prefix}-webui-sg"
  vpc_id = alicloud_vpc.default.id
}

resource "alicloud_security_group_rule" "http" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "8080/8080"
  security_group_id = alicloud_security_group.webui.id
  cidr_ip           = "0.0.0.0/0"
}

resource "alicloud_security_group_rule" "ssh" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "22/22"
  security_group_id = alicloud_security_group.webui.id
  cidr_ip           = "0.0.0.0/0"
}

resource "alicloud_security_group_rule" "egress" {
  type              = "egress"
  ip_protocol       = "all"
  port_range        = "-1/-1"
  security_group_id = alicloud_security_group.webui.id
  cidr_ip           = "0.0.0.0/0"
}

# ECS instance
resource "alicloud_instance" "webui" {
  instance_name        = "${local.prefix}-open-webui"
  instance_type        = "ecs.t6-c1m1.large"  # 2 vCPU, 2 GB — burstable
  image_id             = "ubuntu_22_04_x64_20G_alibase_20260213.vhd"
  security_groups      = [alicloud_security_group.webui.id]
  vswitch_id           = alicloud_vswitch.default.id
  internet_max_bandwidth_out = 10  # 10 Mbps public IP
  system_disk_category = "cloud_efficiency"
  system_disk_size     = 40

  spot_strategy     = "SpotAsPriceGo"  # Cheapest spot pricing
  spot_price_limit  = 0.05  # Max $0.05/hr

  user_data = base64encode(<<-USERDATA
    #!/bin/bash
    set -e
    apt-get update -qq
    apt-get install -y -qq docker.io
    systemctl enable docker
    systemctl start docker

    # Run Open WebUI
    docker run -d \
      --name open-webui \
      --restart unless-stopped \
      -p 8080:8080 \
      -v open-webui-data:/app/backend/data \
      -e OPENAI_API_BASE_URLS="https://raskl-rag-api-rsxcnkcipr.ap-southeast-1.fcapp.run/v1" \
      -e OPENAI_API_KEYS="${var.rag_api_key}" \
      -e ENABLE_OLLAMA_API="false" \
      -e WEBUI_AUTH="true" \
      -e AIOHTTP_CLIENT_TIMEOUT="300" \
      -e ENABLE_EVALUATION_ARENA_MODELS="false" \
      -e ENABLE_WEBSOCKET_SUPPORT="false" \
      -e DATABASE_URL="${local.neon_open_webui_dsn}" \
      -e WEBUI_NAME="SwetBot" \
      ghcr.io/open-webui/open-webui:main
  USERDATA
  )

  tags = {
    Name = "raskl-open-webui"
  }
}

output "webui_public_ip" {
  description = "Open WebUI ECS public IP"
  value       = alicloud_instance.webui.public_ip
}
