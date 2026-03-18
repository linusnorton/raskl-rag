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
  port_range        = "80/80"
  security_group_id = alicloud_security_group.webui.id
  cidr_ip           = "0.0.0.0/0"
}

resource "alicloud_security_group_rule" "https" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "443/443"
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
    export DEBIAN_FRONTEND=noninteractive

    apt-get update -qq
    apt-get install -y -qq docker.io nginx certbot python3-certbot-nginx
    systemctl enable docker nginx
    systemctl start docker

    # Run Open WebUI on localhost:8080
    docker run -d \
      --name open-webui \
      --restart unless-stopped \
      -p 127.0.0.1:8080:8080 \
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

    # Nginx reverse proxy config
    cat > /etc/nginx/sites-available/swetbot2 << 'NGINX'
    server {
        listen 80;
        server_name swetbot2.ljn.io;

        location / {
            proxy_pass http://127.0.0.1:8080;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 300s;
        }
    }
    NGINX

    ln -sf /etc/nginx/sites-available/swetbot2 /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl reload nginx

    # Wait for Open WebUI to be ready, then get SSL cert
    sleep 30
    certbot --nginx -d swetbot2.ljn.io --non-interactive --agree-tos -m linus@ljn.io --redirect

    # Auto-renew cron
    echo "0 3 * * * certbot renew --quiet" | crontab -
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
