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
