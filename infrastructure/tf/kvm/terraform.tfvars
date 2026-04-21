suffix      = "spendsense"
key         = "mlops_proj20"
reservation = "ca122638-97d8-49b5-9955-8cac9e0d4ae8"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "mlops_proj20_devops"
floating_ip_pool = "public"
