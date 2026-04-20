# SpendSense Chameleon Automation

This directory follows the same broad Day 0 / Day 1 layout used in the
Teaching on Testbeds MLOps materials:

- `../tf/kvm`: Terraform provisions Chameleon/OpenStack resources.
- `pre_k8s`: base OS prep on provisioned nodes.
- `k8s`: Kubernetes installation and cluster bootstrap.
- `post_k8s`: post-install add-ons and cluster bootstrap tasks.
- `argocd`: optional GitOps bootstrap for Argo CD.

## Expected workflow

1. Copy `../tf/kvm/terraform.tfvars.example` to `terraform.tfvars` and fill the 5 Chameleon placeholders plus `keypair_name`.
2. Run Terraform from `../tf/kvm`.
3. Terraform generates `inventory/chameleon/hosts.yml`.
4. Copy `group_vars/all.yml.example` to `group_vars/all.yml` and set `k3s_token`.
5. Run:

```bash
ansible-playbook pre_k8s/site.yml
ansible-playbook k8s/site.yml
ansible-playbook post_k8s/site.yml
ansible-playbook argocd/site.yml
```

## Chameleon placeholders you must set

- `image_name`
- `control_plane_flavor`
- `worker_flavor`
- `external_network_name`
- `floating_ip_pool`
- `keypair_name`
- `k3s_token` in `group_vars/all.yml`

Everything else can stay at the defaults unless your Chameleon setup requires a change.
