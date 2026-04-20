openstack_cloud = "openstack"
cluster_name    = "spendsense"
keypair_name    = "REPLACE_WITH_CHAMELEON_KEYPAIR"
ssh_user        = "cc"

# Fill only these Chameleon-specific placeholders.
image_name            = "CC-Ubuntu24.04"
control_plane_flavor  = "m1.large"
worker_flavor         = "m1.large"
external_network_name = "REPLACE_WITH_ACTUAL_EXTERNAL_NETWORK"
reservation_id        = "8b675bdd-33eb-4288-8a5d-bfa3959add5a"

worker_count = 0

# Optional. If omitted, Terraform uses external_network_name as the floating IP pool.
# floating_ip_pool = "REPLACE_WITH_FLOATING_IP_POOL"

# Optional: uncomment only if you need custom networking or tighter access.
# network_name = "spendsense-net"
# subnet_name  = "spendsense-subnet"
# subnet_cidr  = "192.168.42.0/24"
# router_name  = "spendsense-router"
#
# ssh_allowed_cidrs      = ["0.0.0.0/0"]
# api_allowed_cidrs      = ["0.0.0.0/0"]
# http_allowed_cidrs     = ["0.0.0.0/0"]
# nodeport_allowed_cidrs = ["0.0.0.0/0"]
