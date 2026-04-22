suffix      = "proj20"
key         = "mlops_proj20"
reservation = "ee9cd5a0-8d14-4f2e-b140-c143ac43c5fb"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "mlops_proj20_devops"
floating_ip_pool = "public"
