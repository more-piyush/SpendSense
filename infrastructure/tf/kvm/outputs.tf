output "control_plane_public_ip" {
  value       = openstack_networking_floatingip_v2.control_plane.address
  description = "Floating IP of the primary control-plane node."
}

output "worker_public_ips" {
  value       = [for fip in openstack_networking_floatingip_v2.workers : fip.address]
  description = "Floating IPs of worker nodes."
}

output "private_network_id" {
  value       = openstack_networking_network_v2.cluster.id
  description = "Private network ID created for the cluster."
}

output "ansible_inventory_path" {
  value       = local_file.ansible_inventory.filename
  description = "Path to the generated Ansible inventory file."
}

output "floating_ip_pool_used" {
  value       = local.effective_floating_ip_pool
  description = "Floating IP pool Terraform used for allocating floating IPs."
}

output "external_network_used" {
  value       = local.resolved_external_network_name
  description = "External network Terraform used for router egress and floating IP allocation."
}

output "attached_data_volume_id" {
  value       = var.data_volume_name != "" ? data.openstack_blockstorage_volume_v3.data_volume[0].id : null
  description = "Existing Cinder volume ID attached to the control-plane node, if configured."
}
