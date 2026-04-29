suffix      = "proj20"
key         = "mlops_proj20"
reservation = "02a76ede-9972-4546-b40f-0dd83c721376"

ssh_user   = "cc"
image_name = "CC-Ubuntu24.04"

nodes = {
  node1 = "192.168.1.11"
}

data_volume_name = "proj20_devops"
floating_ip_pool = "public"
