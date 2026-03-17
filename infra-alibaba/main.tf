terraform {
  required_version = ">= 1.5"

  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230"
    }
    neon = {
      source  = "kislerdm/neon"
      version = "~> 0.6"
    }
  }

  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "alicloud" {
  region     = var.region
  access_key = var.access_key_id
  secret_key = var.access_key_secret
}

provider "neon" {
  api_key = var.neon_api_key
}

data "alicloud_account" "current" {}

locals {
  account_id = data.alicloud_account.current.id
  region     = var.region
  prefix     = "raskl"
}
