variable "suffix" {
  description = "Suffix for resource names (use net ID or project suffix)."
  type        = string
  nullable    = false
}

variable "key" {
  description = "Name of key pair."
  type        = string
  default     = "id_rsa_chameleon"
}

variable "reservation" {
  description = "UUID of the reserved flavor from the Chameleon lease."
  type        = string
}

variable "nodes" {
  description = "Map of node names to private IPs on the private network."
  type        = map(string)
  default = {
    node1 = "192.168.1.11"
  }
}

variable "ssh_user" {
  description = "SSH username for the chosen image."
  type        = string
  default     = "cc"
}

variable "image_name" {
  description = "OpenStack image name for all nodes."
  type        = string
  default     = "CC-Ubuntu24.04"
}

variable "data_volume_name" {
  description = "Optional existing Cinder volume name to attach to node1."
  type        = string
  default     = ""
}

variable "floating_ip_pool" {
  description = "Floating IP pool/network name for node1 public access."
  type        = string
  default     = "public"
}

variable "public_ingress_cidr" {
  description = "CIDR allowed to access the public-facing SSH and NodePort services."
  type        = string
  default     = "0.0.0.0/0"
}

variable "public_tcp_ports" {
  description = "Public TCP ports to open on the control-plane floating IP."
  type        = list(number)
  default = [
    22,
    30080,
    30081,
    30300,
    30500,
    30900,
    30901,
  ]
}
