# Firefly III DevOps / Platform — Initial Implementation

This repository is a complete initial implementation for the DevOps / Platform role.

## Included components

- Firefly III
- PostgreSQL with persistence
- MinIO with persistence
- MLflow
- Dummy inference API
- Dummy training job
- Nightly evaluation CronJob
- Monthly retrain CronJob
- Namespace, services, PVCs, scripts, and documentation

## Before running

Edit these placeholders:
- `<DB_PASSWORD>`
- `<FIREFLY_APP_KEY>`
- `<FIREFLY_URL>`
- `<MINIO_ROOT_USER>`
- `<MINIO_ROOT_PASSWORD>`
- `<NODE_IP>`

## Suggested run order

```bash
cd infrastructure
chmod +x scripts/*.sh
./scripts/bootstrap.sh
./scripts/create-secrets.sh
./scripts/deploy-core.sh
./scripts/validate-core.sh
```

## Browser / UI access

- Firefly III: `http://<NODE_IP>:30080`
- MLflow: `kubectl port-forward svc/mlflow 5000:5000 -n firefly-platform`
- MinIO Console: `kubectl port-forward svc/minio 9001:9001 -n firefly-platform`

## Teardown

```bash
./scripts/teardown-core.sh
```

## Notes

- `bootstrap.sh` is intentionally light and assumes you already have Kubernetes / K3s available from your lab setup.
- This is optimized for the initial implementation milestone, not the final integrated system.
