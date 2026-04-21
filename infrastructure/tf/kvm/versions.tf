terraform {
  required_version = ">= 1.5.0"

  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.54"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}
