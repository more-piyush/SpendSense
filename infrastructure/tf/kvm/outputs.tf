output "floating_ip_out" {
  description = "Floating IP assigned to node1."
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "control_plane_public_ip" {
  description = "Floating IP assigned to the control-plane node."
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "private_network_id" {
  description = "Private network ID created for the cluster."
  value       = openstack_networking_network_v2.private_net.id
}

output "ansible_inventory_path" {
  description = "Path to the generated Ansible inventory file."
  value       = local_file.ansible_inventory.filename
}

output "attached_data_volume_id" {
  description = "Existing Cinder volume ID attached to node1, if configured."
  value       = var.data_volume_name != "" ? data.openstack_blockstorage_volume_v3.data_volume[0].id : null
}
