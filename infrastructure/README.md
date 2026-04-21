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

## Single-VM full deployment

For a manually created Ubuntu VM with an attached data volume, the repository now
includes a one-command bootstrap that:

- updates the OS
- installs Docker, K3s, Helm, Ansible, and Terraform
- mounts and formats the attached volume if needed
- builds and imports all required application images
- deploys the core platform, raw-data seed, one-off data bootstrap, initial training, active-model selection, serving restart, monitoring, and validation

Preparation:

```bash
cd infrastructure
cp config/deploy.env.example config/deploy.env
```

Edit `config/deploy.env` and set at minimum:

- `FLOATING_IP`
- `DATA_VOLUME_DEVICE`
- `MINIO_ROOT_USER` if you do not want the default
- any secrets you do not want auto-generated

By default, the deployment keeps the root disk for Ubuntu, packages, logs, and
normal system files, and uses the attached block volume for heavy runtime data:

- Docker data root: `/data/docker`
- K3s/containerd data: `/data/k3s`
- Kubernetes local-path PVC storage: `/data/local-path-provisioner`
- temporary build/cache files: `/data/tmp`

Then run from the repository root:

```bash
bash infrastructure/scripts/deploy-all.sh infrastructure/config/deploy.env
```

The deployment writes a timestamped log under `infrastructure/logs/` and prints
phase-by-phase status as it runs. Initial training jobs are launched in
parallel; after the configured warmup, the script attempts active-model
selection and restarts the serving deployment once models are ready.

Run that command from a normal user with `sudo` access rather than from a
fully root-owned shell so the kubeconfig and Docker group setup land on the
intended operator account.

All infrastructure scripts now read the same env file by default:

```bash
infrastructure/config/deploy.env
```

That means you can also run individual steps like these without re-exporting
variables:

```bash
bash infrastructure/scripts/create-secrets-from-env.sh
bash infrastructure/scripts/deploy-core.sh
bash infrastructure/scripts/run-initial-training.sh
bash infrastructure/scripts/deploy-monitoring.sh
bash infrastructure/scripts/validate-core.sh
```

## GitHub Actions deployment

The repository also includes a manual GitHub Actions workflow at
`.github/workflows/deploy-infrastructure.yml` for the Chameleon
Terraform + Ansible + application deployment path.

It performs these phases:

- renders `terraform.tfvars`, `group_vars/all.yml`, and `config/deploy.env`
- runs Terraform to provision the nodes and floating IP
- runs Ansible `pre_k8s`, `k8s`, `post_k8s`, and optionally `argocd`
- syncs the repository to the control-plane node
- runs `infrastructure/scripts/deploy-remote-stack.sh` on the control plane to build images, deploy the platform, bootstrap data, start initial training, activate models, restart serving, deploy monitoring, and validate the stack

Required GitHub secrets and variables:

- Secret: `OPENSTACK_CLOUDS_YAML`
- Secret: `CHAMELEON_SSH_PRIVATE_KEY`
- Secret: `TF_RESERVATION_ID`
- Secret: `K3S_TOKEN`
- Secret: `POSTGRES_PASSWORD`
- Secret: `MINIO_ROOT_PASSWORD`
- Variable: `TF_SUFFIX`
- Variable: `TF_KEYPAIR_NAME`
- Variable: `TF_FLOATING_IP_POOL`
- Variable: `ANSIBLE_SSH_USER`
- Variable: `DATA_VOLUME_DEVICE`
- Variable: `DATA_VOLUME_MOUNT_PATH`
- Variable: `DATA_VOLUME_FILESYSTEM`
- Variable: `DOCKER_DATA_ROOT`
- Variable: `K3S_DATA_DIR`
- Variable: `K3S_LOCAL_STORAGE_PATH`
- Variable: `BUILD_CACHE_ROOT`

Optional variables:

- `TF_IMAGE_NAME`
- `TF_DATA_VOLUME_NAME`
- `TF_NODES_YAML`
- `ARGOCD_REPO_URL`
- `ARGOCD_TARGET_REVISION`
- `INITIAL_TRAINING_WARMUP_SECONDS`
- `ACTIVE_MODEL_RETRY_SECONDS`
- `ACTIVE_MODEL_MAX_WAIT_SECONDS`

`FIREFLY_APP_KEY` is optional in both the local env file and the GitHub Actions
workflow. If you leave it blank, the deployment flow generates a valid Firefly
APP key automatically.

Run it from the Actions tab with `workflow_dispatch`.

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

## Monitoring

Deploy the monitoring stack after the core platform is running:

```bash
./scripts/deploy-monitoring.sh
./scripts/validate-monitoring.sh
```

Grafana is exposed on NodePort `30300` by default.

## Teardown

```bash
./scripts/teardown-core.sh
```

## Notes

- Deployment uses internal Kubernetes DNS for service-to-service communication.
- `deploy-core.sh` renders Firefly `APP_URL` from `FLOATING_IP`.
- `seed-data.sh` uploads `Data/intrvw24.zip` and `Data/CE-HG-Integ-2024.txt` into `processed-data/raw/`.
