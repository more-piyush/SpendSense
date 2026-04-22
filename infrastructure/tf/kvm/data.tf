data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  name = "sharednet1-subnet"
}

data "openstack_blockstorage_volume_v3" "data_volume" {
  count = var.data_volume_name != "" ? 1 : 0
  name  = var.data_volume_name
}
