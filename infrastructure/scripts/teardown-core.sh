#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../k8s"

kubectl delete -f "${K8S_DIR}/cronjobs/monthly-retrain.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/cronjobs/nightly-eval.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/training/configmap.yaml" --ignore-not-found=true

kubectl delete -f "${K8S_DIR}/data/cronjob.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/data/configmap.yaml" --ignore-not-found=true

kubectl delete -f "${K8S_DIR}/serving/service.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/serving/deployment.yaml" --ignore-not-found=true

kubectl delete -f "${K8S_DIR}/firefly/service.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/firefly/deployment.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/firefly/configmap.yaml" --ignore-not-found=true

kubectl delete -f "${K8S_DIR}/postgres-bootstrap/job.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/minio-bootstrap/job.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/mlflow/service.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/mlflow/deployment.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/minio/service.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/minio/deployment.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/postgres/service.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/postgres/deployment.yaml" --ignore-not-found=true

echo "[INFO] PVCs and namespace preserved by default."
