variable "openstack_cloud" {
  description = "Cloud entry name from clouds.yaml."
  type        = string
  default     = "openstack"
}

variable "cluster_name" {
  description = "Prefix used for all Chameleon resources."
  type        = string
  default     = "spendsense"
}

variable "keypair_name" {
  description = "Existing OpenStack keypair name uploaded to Chameleon."
  type        = string
}

variable "ssh_user" {
  description = "SSH username for the chosen image."
  type        = string
  default     = "cc"
}

variable "image_name" {
  description = "OpenStack image name for all nodes."
  type        = string
}

variable "control_plane_flavor" {
  description = "Flavor for the control-plane instance."
  type        = string
}

variable "worker_flavor" {
  description = "Flavor for worker instances."
  type        = string
}

variable "worker_count" {
  description = "Number of worker nodes to provision."
  type        = number
  default     = 2
}

variable "reservation_id" {
  description = "Existing Chameleon reservation UUID for the KVM flavor lease."
  type        = string
  default     = ""
}

variable "data_volume_name" {
  description = "Optional existing Cinder volume name to attach to the control-plane node."
  type        = string
  default     = ""
}

variable "network_name" {
  description = "Private network name to create for the cluster."
  type        = string
  default     = "spendsense-net"
}

variable "subnet_name" {
  description = "Private subnet name to create for the cluster."
  type        = string
  default     = "spendsense-subnet"
}

variable "subnet_cidr" {
  description = "CIDR for the private subnet."
  type        = string
  default     = "192.168.42.0/24"
}

variable "router_name" {
  description = "Router name used to reach the external network."
  type        = string
  default     = "spendsense-router"
}

variable "external_network_name" {
  description = "Optional external network name in OpenStack. Leave empty to auto-discover the external network."
  type        = string
  default     = ""
}

variable "floating_ip_pool" {
  description = "Floating IP pool/network name for node access. Leave empty to reuse external_network_name."
  type        = string
  default     = ""
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to SSH to the cluster."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "api_allowed_cidrs" {
  description = "CIDRs allowed to reach the Kubernetes API."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "http_allowed_cidrs" {
  description = "CIDRs allowed to reach HTTP/HTTPS ingress."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "nodeport_allowed_cidrs" {
  description = "CIDRs allowed to reach NodePort services."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "create_security_group" {
  description = "Whether Terraform should create a dedicated security group."
  type        = bool
  default     = true
}
