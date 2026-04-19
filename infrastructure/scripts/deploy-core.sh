#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FLOATING_IP:-}" ]]; then
  echo "[ERROR] FLOATING_IP is required. Example: export FLOATING_IP=129.114.x.x"
  exit 1
fi

kubectl apply -f ../k8s/namespace/namespace.yaml
kubectl apply -f ../k8s/storage/pvc-postgres.yaml
kubectl apply -f ../k8s/storage/pvc-minio.yaml

kubectl apply -f ../k8s/postgres/deployment.yaml
kubectl apply -f ../k8s/postgres/service.yaml

kubectl apply -f ../k8s/minio/deployment.yaml
kubectl apply -f ../k8s/minio/service.yaml

kubectl apply -f ../k8s/mlflow/deployment.yaml
kubectl apply -f ../k8s/mlflow/service.yaml

kubectl apply -f ../k8s/minio-bootstrap/job.yaml

tmp_config="$(mktemp)"
sed "s|<FLOATING_IP>|${FLOATING_IP}|g" ../k8s/firefly/configmap.yaml > "$tmp_config"
kubectl apply -f "$tmp_config"
rm -f "$tmp_config"

kubectl apply -f ../k8s/firefly/deployment.yaml
kubectl apply -f ../k8s/firefly/service.yaml

kubectl apply -f ../k8s/serving/deployment.yaml
kubectl apply -f ../k8s/serving/service.yaml

kubectl apply -f ../k8s/data/configmap.yaml
kubectl apply -f ../k8s/data/cronjob.yaml

kubectl apply -f ../k8s/training/configmap.yaml
kubectl apply -f ../k8s/cronjobs/nightly-eval.yaml
kubectl apply -f ../k8s/cronjobs/monthly-retrain.yaml

echo "[INFO] Integrated core services deployed."
