#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../k8s"

if [[ -z "${FLOATING_IP:-}" ]]; then
  echo "[ERROR] FLOATING_IP is required. Example: export FLOATING_IP=129.114.x.x"
  exit 1
fi

kubectl apply -f "${K8S_DIR}/namespace/namespace.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-postgres.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-minio.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-training-state.yaml"

kubectl apply -f "${K8S_DIR}/postgres/deployment.yaml"
kubectl apply -f "${K8S_DIR}/postgres/service.yaml"
kubectl apply -f "${K8S_DIR}/postgres-bootstrap/job.yaml"

kubectl apply -f "${K8S_DIR}/minio/deployment.yaml"
kubectl apply -f "${K8S_DIR}/minio/service.yaml"

kubectl apply -f "${K8S_DIR}/mlflow/deployment.yaml"
kubectl apply -f "${K8S_DIR}/mlflow/service.yaml"

kubectl apply -f "${K8S_DIR}/minio-bootstrap/job.yaml"

tmp_config="$(mktemp)"
sed "s|<FLOATING_IP>|${FLOATING_IP}|g" "${K8S_DIR}/firefly/configmap.yaml" > "$tmp_config"
kubectl apply -f "$tmp_config"
rm -f "$tmp_config"

kubectl apply -f "${K8S_DIR}/firefly/deployment.yaml"
kubectl apply -f "${K8S_DIR}/firefly/service.yaml"

kubectl apply -f "${K8S_DIR}/serving/deployment.yaml"
kubectl apply -f "${K8S_DIR}/serving/service.yaml"

kubectl apply -f "${K8S_DIR}/data/configmap.yaml"
kubectl apply -f "${K8S_DIR}/data/cronjob.yaml"

kubectl apply -f "${K8S_DIR}/training/configmap.yaml"
kubectl apply -f "${K8S_DIR}/cronjobs/nightly-eval.yaml"
kubectl apply -f "${K8S_DIR}/cronjobs/monthly-retrain.yaml"

echo "[INFO] Integrated core services deployed."
