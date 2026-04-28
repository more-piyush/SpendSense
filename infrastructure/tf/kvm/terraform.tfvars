suffix      = "proj20"
key         = "mlops_proj20"
reservation = "582193d1-60a8-44a6-9686-ea4076c21ffc"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "mlops_proj20_devops"
floating_ip_pool = "public"
