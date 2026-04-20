locals {
  control_plane_name         = "${var.cluster_name}-cp-1"
  worker_names               = [for i in range(var.worker_count) : format("%s-worker-%02d", var.cluster_name, i + 1)]
  effective_floating_ip_pool = var.floating_ip_pool != "" ? var.floating_ip_pool : var.external_network_name
}

resource "openstack_networking_network_v2" "cluster" {
  name           = var.network_name
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "cluster" {
  name            = var.subnet_name
  network_id      = openstack_networking_network_v2.cluster.id
  cidr            = var.subnet_cidr
  ip_version      = 4
  dns_nameservers = ["8.8.8.8", "1.1.1.1"]
}

data "openstack_networking_network_v2" "external" {
  name = var.external_network_name
}

resource "openstack_networking_router_v2" "cluster" {
  name                = var.router_name
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "cluster" {
  router_id = openstack_networking_router_v2.cluster.id
  subnet_id = openstack_networking_subnet_v2.cluster.id
}

resource "openstack_networking_secgroup_v2" "cluster" {
  count       = var.create_security_group ? 1 : 0
  name        = "${var.cluster_name}-sg"
  description = "SpendSense Kubernetes security group"
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  for_each          = var.create_security_group ? toset(var.ssh_allowed_cidrs) : toset([])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = each.value
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "k8s_api" {
  for_each          = var.create_security_group ? toset(var.api_allowed_cidrs) : toset([])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6443
  port_range_max    = 6443
  remote_ip_prefix  = each.value
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "http" {
  for_each          = var.create_security_group ? toset(var.http_allowed_cidrs) : toset([])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = each.value
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "https" {
  for_each          = var.create_security_group ? toset(var.http_allowed_cidrs) : toset([])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 443
  port_range_max    = 443
  remote_ip_prefix  = each.value
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "nodeport" {
  for_each          = var.create_security_group ? toset(var.nodeport_allowed_cidrs) : toset([])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30000
  port_range_max    = 32767
  remote_ip_prefix  = each.value
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "all_internal" {
  count             = var.create_security_group ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  remote_ip_prefix  = var.subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_networking_secgroup_rule_v2" "egress" {
  count             = var.create_security_group ? 1 : 0
  direction         = "egress"
  ethertype         = "IPv4"
  security_group_id = openstack_networking_secgroup_v2.cluster[0].id
}

resource "openstack_compute_instance_v2" "control_plane" {
  name            = local.control_plane_name
  image_name      = var.image_name
  flavor_name     = var.control_plane_flavor
  key_pair        = var.keypair_name
  security_groups = var.create_security_group ? [openstack_networking_secgroup_v2.cluster[0].name] : []

  network {
    uuid = openstack_networking_network_v2.cluster.id
  }

  dynamic "scheduler_hints" {
    for_each = var.reservation_id != "" ? [1] : []
    content {
      additional_properties = {
        reservation = var.reservation_id
      }
    }
  }
}

resource "openstack_networking_floatingip_v2" "control_plane" {
  pool = local.effective_floating_ip_pool
}

resource "openstack_compute_floatingip_associate_v2" "control_plane" {
  floating_ip = openstack_networking_floatingip_v2.control_plane.address
  instance_id = openstack_compute_instance_v2.control_plane.id
}

resource "openstack_compute_instance_v2" "workers" {
  count           = var.worker_count
  name            = local.worker_names[count.index]
  image_name      = var.image_name
  flavor_name     = var.worker_flavor
  key_pair        = var.keypair_name
  security_groups = var.create_security_group ? [openstack_networking_secgroup_v2.cluster[0].name] : []

  network {
    uuid = openstack_networking_network_v2.cluster.id
  }

  dynamic "scheduler_hints" {
    for_each = var.reservation_id != "" ? [1] : []
    content {
      additional_properties = {
        reservation = var.reservation_id
      }
    }
  }
}

resource "openstack_networking_floatingip_v2" "workers" {
  count = var.worker_count
  pool  = local.effective_floating_ip_pool
}

resource "openstack_compute_floatingip_associate_v2" "workers" {
  count       = var.worker_count
  floating_ip = openstack_networking_floatingip_v2.workers[count.index].address
  instance_id = openstack_compute_instance_v2.workers[count.index].id
}

resource "local_file" "ansible_inventory" {
  filename = "${path.module}/../../ansible/inventory/chameleon/hosts.yml"
  content = templatefile("${path.module}/templates/hosts.yml.tftpl", {
    ssh_user           = var.ssh_user
    control_plane_name = openstack_compute_instance_v2.control_plane.name
    control_plane_ip   = openstack_networking_floatingip_v2.control_plane.address
    worker_nodes = [
      for idx, instance in openstack_compute_instance_v2.workers : {
        name = instance.name
        ip   = openstack_networking_floatingip_v2.workers[idx].address
      }
    ]
  })
}
