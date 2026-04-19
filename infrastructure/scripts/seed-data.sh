#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/Data"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://127.0.0.1:30900}"
MINIO_BUCKET="${MINIO_BUCKET:-processed-data}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl is required."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is required."
  exit 1
fi

if [[ ! -f "${DATA_DIR}/intrvw24.zip" ]]; then
  echo "[ERROR] Missing ${DATA_DIR}/intrvw24.zip"
  exit 1
fi

if [[ ! -f "${DATA_DIR}/CE-HG-Integ-2024.txt" ]]; then
  echo "[ERROR] Missing ${DATA_DIR}/CE-HG-Integ-2024.txt"
  exit 1
fi

MINIO_ROOT_USER="$(
  kubectl get secret minio-secret -n firefly-platform \
    -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 --decode
)"
MINIO_ROOT_PASSWORD="$(
  kubectl get secret minio-secret -n firefly-platform \
    -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode
)"

echo "[INFO] Seeding raw data files into ${MINIO_BUCKET}/raw via ${MINIO_ENDPOINT}"

docker run --rm \
  --network host \
  -v "${DATA_DIR}:/seed:ro" \
  minio/mc:latest \
  /bin/sh -c "
    set -e
    mc alias set local '${MINIO_ENDPOINT}' '${MINIO_ROOT_USER}' '${MINIO_ROOT_PASSWORD}'
    mc mb -p local/${MINIO_BUCKET} || true
    mc cp /seed/intrvw24.zip local/${MINIO_BUCKET}/raw/intrvw24.zip
    mc cp /seed/CE-HG-Integ-2024.txt local/${MINIO_BUCKET}/raw/CE-HG-Integ-2024.txt
    mc ls local/${MINIO_BUCKET}/raw
  "

echo "[INFO] Raw data seeded successfully."
