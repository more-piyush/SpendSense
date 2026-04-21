suffix      = "proj20"
key         = "mlops_proj20"
reservation = "c061a217-d565-43d6-a808-c34b604c1f34"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "mlops_proj20_devops"
floating_ip_pool = "public"
