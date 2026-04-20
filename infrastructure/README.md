# Firefly III DevOps / Platform — Integrated Kubernetes Stack

This repository now includes a Chameleon-focused IaC/CaC scaffold in the same
spirit as the Teaching on Testbeds MLOps example:

- `infrastructure/tf/kvm`: Terraform for Chameleon/OpenStack provisioning
- `infrastructure/ansible/pre_k8s`: node bootstrap
- `infrastructure/ansible/k8s`: Kubernetes installation
- `infrastructure/ansible/post_k8s`: cluster bootstrap
- `infrastructure/ansible/argocd`: GitOps bootstrap

This repository now deploys an integrated MLOps stack on Kubernetes with IP-based access (no DNS required).

## Chameleon IaC / CaC workflow

1. Create a `clouds.yaml` with your Chameleon application credential.
2. Fill placeholders in `infrastructure/tf/kvm/terraform.tfvars.example` and save as `terraform.tfvars`.
3. Run Terraform:

```bash
cd infrastructure/tf/kvm
terraform init
terraform plan
terraform apply
```

4. Copy `infrastructure/ansible/group_vars/all.yml.example` to `all.yml` and set values.
5. Run Ansible from `infrastructure/ansible`:

```bash
ansible-playbook pre_k8s/site.yml
ansible-playbook k8s/site.yml
ansible-playbook post_k8s/site.yml
ansible-playbook argocd/site.yml
```

Terraform writes the generated inventory to
`infrastructure/ansible/inventory/chameleon/hosts.yml`.

For the simplest setup, only change:

- `keypair_name`
- `image_name`
- `control_plane_flavor`
- `worker_flavor`
- `external_network_name`
- `floating_ip_pool`
- `reservation_id`
- `k3s_token`

## Included components

- Firefly III (NodePort)
- PostgreSQL with persistence
- MinIO with persistence (API + Console NodePort)
- MLflow (NodePort)
- Serving API (NodePort)
- Data pipeline nightly CronJob
- Training retrain CronJobs (nightly categ check + monthly trend retrain)
- MinIO bucket bootstrap Job

## Before running

1. Create secrets:

```bash
cd infrastructure
chmod +x scripts/*.sh
./scripts/bootstrap.sh
./scripts/create-secrets.sh
```

2. Set your instance floating IP:

```bash
export FLOATING_IP=<YOUR_FLOATING_IP>
```

3. Deploy and validate:

```bash
./scripts/deploy-core.sh
./scripts/validate-core.sh
```

4. Seed raw data required by the data pipeline:

```bash
./scripts/seed-data.sh
```

## External access (Floating IP)

- Firefly III: `http://<FLOATING_IP>:30080`
- Serving API: `http://<FLOATING_IP>:30081`
- MLflow UI: `http://<FLOATING_IP>:30500`
- MinIO API: `http://<FLOATING_IP>:30900`
- MinIO Console: `http://<FLOATING_IP>:30901`

## Container images required

Build and push these images before production use:

- `firefly-data:latest` (from `Data/pipelines/Dockerfile`)
- `spendsense/training:latest` (from `Training/training_scripts/Dockerfile`)

Serving currently uses: `spendsense/serving-unified:latest`.
The unified serving stack loads the active MLflow artifacts from `active_models.json`;
the older ONNX example images under `Serving/` are not used by the production deployment.

## Teardown

```bash
./scripts/teardown-core.sh
```

## Notes

- Deployment uses internal Kubernetes DNS for service-to-service communication.
- `deploy-core.sh` renders Firefly `APP_URL` from `FLOATING_IP`.
- `seed-data.sh` uploads `Data/intrvw24.zip` and `Data/CE-HG-Integ-2024.txt` into `processed-data/raw/`.
