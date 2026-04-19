# Firefly III DevOps / Platform — Integrated Kubernetes Stack

This repository now deploys an integrated MLOps stack on Kubernetes with IP-based access (no DNS required).

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

## External access (Floating IP)

- Firefly III: `http://<FLOATING_IP>:30080`
- Serving API: `http://<FLOATING_IP>:30081`
- MLflow UI: `http://<FLOATING_IP>:30500`
- MinIO API: `http://<FLOATING_IP>:30900`
- MinIO Console: `http://<FLOATING_IP>:30901`

## Container images required

Build and push these images before production use:

- `spendsense/data-pipeline:latest` (from `Data/pipelines/Dockerfile.pipeline`)
- `spendsense/training:latest` (from `Training/training_scripts/Dockerfile`)

Serving currently uses: `pranalithakkar/serving-baseline:latest`.

## Teardown

```bash
./scripts/teardown-core.sh
```

## Notes

- Deployment uses internal Kubernetes DNS for service-to-service communication.
- `deploy-core.sh` renders Firefly `APP_URL` from `FLOATING_IP`.
