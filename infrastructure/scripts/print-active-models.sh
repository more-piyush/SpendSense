#!/usr/bin/env bash
set -euo pipefail

ACTIVE_FILE="${1:-active_models.json}"
NS="firefly-platform"
TRAINING_IMAGE="${TRAINING_IMAGE:-spendsense/training:latest}"
POD_NAME="active-models-check-$(date +%s)"

kubectl run "${POD_NAME}" \
  -n "${NS}" \
  --restart=Never \
  --image="${TRAINING_IMAGE}" \
  --image-pull-policy=IfNotPresent \
  --env="AWS_ACCESS_KEY_ID=$(kubectl get secret minio-secret -n "${NS}" -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 -d)" \
  --env="AWS_SECRET_ACCESS_KEY=$(kubectl get secret minio-secret -n "${NS}" -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 -d)" \
  --env="MLFLOW_S3_ENDPOINT_URL=http://minio.firefly-platform.svc.cluster.local:9000" \
  --command -- \
  python -c "from utils import load_active_models; import json; print(json.dumps(load_active_models('s3://mlflow/registry', active_models_filename='${ACTIVE_FILE}'), indent=2))" >/dev/null

kubectl wait --for=condition=Ready -n "${NS}" "pod/${POD_NAME}" --timeout=5m || true
kubectl logs -n "${NS}" "${POD_NAME}"
kubectl delete pod -n "${NS}" "${POD_NAME}" --ignore-not-found=true >/dev/null
