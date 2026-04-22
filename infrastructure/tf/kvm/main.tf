locals {
  node_keys         = sort(keys(var.nodes))
  control_plane_key = local.node_keys[0]
  worker_keys       = slice(local.node_keys, 1, length(local.node_keys))
}

resource "openstack_networking_secgroup_v2" "public_services" {
  name        = "spendsense-public-${var.suffix}"
  description = "Public SSH and SpendSense NodePort access for ${var.suffix}"
}

resource "openstack_networking_secgroup_rule_v2" "public_tcp_ingress" {
  for_each          = toset([for port in var.public_tcp_ports : tostring(port)])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = tonumber(each.value)
  port_range_max    = tonumber(each.value)
  remote_ip_prefix  = var.public_ingress_cidr
  security_group_id = openstack_networking_secgroup_v2.public_services.id
}

resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-mlops-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-mlops-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
  name       = "sharednet1-${each.key}-mlops-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id
  security_group_ids = [
    openstack_networking_secgroup_v2.public_services.id,
  ]
}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name       = "${each.key}-mlops-${var.suffix}"
  image_name = var.image_name
  flavor_id  = var.reservation
  key_pair   = var.key

  network {
    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = var.floating_ip_pool
  description = "SpendSense IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet1_ports[local.control_plane_key].id
}

resource "openstack_compute_volume_attach_v2" "control_plane_data_volume" {
  count       = var.data_volume_name != "" ? 1 : 0
  instance_id = openstack_compute_instance_v2.nodes[local.control_plane_key].id
  volume_id   = data.openstack_blockstorage_volume_v3.data_volume[0].id
}

resource "local_file" "ansible_inventory" {
  filename = "${path.module}/../../ansible/inventory/chameleon/hosts.yml"
  content = templatefile("${path.module}/templates/hosts.yml.tftpl", {
    ssh_user           = var.ssh_user
    control_plane_name = openstack_compute_instance_v2.nodes[local.control_plane_key].name
    control_plane_ip   = openstack_networking_floatingip_v2.floating_ip.address
    worker_nodes = [
      for key in local.worker_keys : {
        name = openstack_compute_instance_v2.nodes[key].name
        ip   = openstack_networking_port_v2.private_net_ports[key].all_fixed_ips[0]
      }
    ]
  })
}
