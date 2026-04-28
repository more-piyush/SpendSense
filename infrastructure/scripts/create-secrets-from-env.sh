#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${KUBECONFIG:-}" && -f "${HOME}/.kube/config" ]]; then
  export KUBECONFIG="${HOME}/.kube/config"
fi

POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
FIREFLY_APP_KEY="${FIREFLY_APP_KEY:-}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-}"

if [[ -z "${POSTGRES_PASSWORD}" ]]; then
  POSTGRES_PASSWORD="$(openssl rand -base64 24 | tr -d '\n')"
fi

if [[ -z "${FIREFLY_APP_KEY}" ]]; then
  FIREFLY_APP_KEY="base64:$(openssl rand -base64 32 | tr -d '\n')"
fi

if [[ -z "${MINIO_ROOT_PASSWORD}" ]]; then
  MINIO_ROOT_PASSWORD="$(openssl rand -base64 24 | tr -d '\n')"
fi

kubectl apply -f "${REPO_ROOT}/infrastructure/k8s/namespace/namespace.yaml"

kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_DB=firefly \
  --from-literal=POSTGRES_USER=firefly \
  --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -n firefly-platform \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic firefly-secret \
  --from-literal=APP_KEY="${FIREFLY_APP_KEY}" \
  -n firefly-platform \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic minio-secret \
  --from-literal=MINIO_ROOT_USER="${MINIO_ROOT_USER}" \
  --from-literal=MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}" \
  -n firefly-platform \
  --dry-run=client -o yaml | kubectl apply -f -

cat <<EOF
[INFO] Secrets are ready.
[INFO] MINIO_ROOT_USER=${MINIO_ROOT_USER}
[INFO] PostgreSQL password configured.
[INFO] Firefly APP_KEY configured.
[INFO] MinIO password configured.
EOF
