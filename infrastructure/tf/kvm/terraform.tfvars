suffix      = "proj20"
key         = "mlops_proj20"
reservation = "3ead04ff-76e8-4d3d-9cc2-97a4c7dfa7ec"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "mlops_proj20_devops"
floating_ip_pool = "public"
