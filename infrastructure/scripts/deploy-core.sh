#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"
K8S_DIR="${SCRIPT_DIR}/../k8s"

if [[ -z "${FLOATING_IP:-}" ]]; then
  echo "[ERROR] FLOATING_IP is required. Set it in infrastructure/config/deploy.env or export it."
  exit 1
fi

kubectl apply -f "${K8S_DIR}/namespace/namespace.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-postgres.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-minio.yaml"
kubectl apply -f "${K8S_DIR}/storage/pvc-training-state.yaml"

kubectl apply -f "${K8S_DIR}/postgres/deployment.yaml"
kubectl apply -f "${K8S_DIR}/postgres/service.yaml"
kubectl rollout status deployment/postgres -n firefly-platform --timeout=240s
kubectl delete -f "${K8S_DIR}/postgres-bootstrap/job.yaml" --ignore-not-found=true
kubectl apply -f "${K8S_DIR}/postgres-bootstrap/job.yaml"
if ! kubectl wait --for=condition=complete job/postgres-bootstrap -n firefly-platform --timeout=240s; then
  echo "[WARN] postgres-bootstrap did not complete on the first attempt."
  bootstrap_logs="$(kubectl logs -n firefly-platform job/postgres-bootstrap 2>&1 || true)"
  postgres_logs="$(kubectl logs -n firefly-platform deployment/postgres 2>&1 || true)"

  if printf '%s\n%s\n' "${bootstrap_logs}" "${postgres_logs}" | grep -qi 'password authentication failed'; then
    echo "[WARN] Detected PostgreSQL password mismatch on existing data volume. Resetting PostgreSQL state."
    kubectl delete -f "${K8S_DIR}/postgres-bootstrap/job.yaml" --ignore-not-found=true
    kubectl delete deployment postgres -n firefly-platform --ignore-not-found=true
    kubectl delete pvc postgres-pvc -n firefly-platform --ignore-not-found=true

    kubectl apply -f "${K8S_DIR}/storage/pvc-postgres.yaml"
    kubectl apply -f "${K8S_DIR}/postgres/deployment.yaml"
    kubectl apply -f "${K8S_DIR}/postgres/service.yaml"
    kubectl rollout status deployment/postgres -n firefly-platform --timeout=240s
    kubectl apply -f "${K8S_DIR}/postgres-bootstrap/job.yaml"
    kubectl wait --for=condition=complete job/postgres-bootstrap -n firefly-platform --timeout=240s
  else
    printf '%s\n' "${bootstrap_logs}"
    printf '%s\n' "${postgres_logs}"
    exit 1
  fi
fi

kubectl apply -f "${K8S_DIR}/minio/deployment.yaml"
kubectl apply -f "${K8S_DIR}/minio/service.yaml"
kubectl rollout status deployment/minio -n firefly-platform --timeout=240s

kubectl apply -f "${K8S_DIR}/mlflow/deployment.yaml"
kubectl apply -f "${K8S_DIR}/mlflow/service.yaml"
kubectl rollout status deployment/mlflow -n firefly-platform --timeout=240s

kubectl delete -f "${K8S_DIR}/minio-bootstrap/job.yaml" --ignore-not-found=true
kubectl apply -f "${K8S_DIR}/minio-bootstrap/job.yaml"
kubectl wait --for=condition=complete job/minio-bootstrap -n firefly-platform --timeout=240s

tmp_config="$(mktemp)"
sed "s|<FLOATING_IP>|${FLOATING_IP}|g" "${K8S_DIR}/firefly/configmap.yaml" > "$tmp_config"
kubectl apply -f "$tmp_config"
rm -f "$tmp_config"

kubectl delete -f "${K8S_DIR}/firefly-bootstrap/job.yaml" --ignore-not-found=true
kubectl apply -f "${K8S_DIR}/firefly-bootstrap/job.yaml"
kubectl wait --for=condition=complete job/firefly-bootstrap -n firefly-platform --timeout=300s

kubectl apply -f "${K8S_DIR}/firefly/deployment.yaml"
kubectl apply -f "${K8S_DIR}/firefly/service.yaml"

kubectl apply -f "${K8S_DIR}/serving/deployment.yaml"
kubectl apply -f "${K8S_DIR}/serving/service.yaml"
kubectl apply -f "${K8S_DIR}/serving/canary-deployment.yaml"
kubectl apply -f "${K8S_DIR}/serving/canary-service.yaml"
kubectl apply -f "${K8S_DIR}/serving/canary-public-service.yaml"

kubectl apply -f "${K8S_DIR}/data/configmap.yaml"
kubectl apply -f "${K8S_DIR}/data/cronjob.yaml"

kubectl apply -f "${K8S_DIR}/training/candidate-configmap.yaml"
kubectl apply -f "${K8S_DIR}/training/configmap.yaml"
kubectl delete -f "${K8S_DIR}/retraining/configmap.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/cronjobs/nightly-eval.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/cronjobs/monthly-retrain.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/cronjobs/weekly-categorization-retraining-data.yaml" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/cronjobs/monthly-trend-retraining-data.yaml" --ignore-not-found=true
kubectl apply -f "${K8S_DIR}/cronjobs/cyclic-categorization-retrain.yaml"
kubectl apply -f "${K8S_DIR}/cronjobs/cyclic-trend-retrain.yaml"

echo "[INFO] Integrated core services deployed."
