#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/Data"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://127.0.0.1:30900}"
MINIO_BUCKET="${MINIO_BUCKET:-processed-data}"

RAW_FILES=(
  "intrvw20.zip"
  "intrvw21.zip"
  "intrvw22.zip"
  "intrvw23 (1).zip"
  "intrvw24.zip"
  "CE-HG-Integ-2020.txt"
  "CE-HG-Integ-2021.txt"
  "CE-HG-Integ-2022.txt"
  "CE-HG-Integ-2023.txt"
  "CE-HG-Integ-2024.txt"
)

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl is required."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  if [[ -z "${DOCKER_CMD:-}" ]]; then
    echo "[ERROR] docker is required."
    exit 1
  fi
fi

DOCKER_CMD="${DOCKER_CMD:-docker}"

if ! command -v "${DOCKER_CMD%% *}" >/dev/null 2>&1; then
  echo "[ERROR] ${DOCKER_CMD} is not available."
  exit 1
fi

for raw_file in "${RAW_FILES[@]}"; do
  if [[ ! -f "${DATA_DIR}/${raw_file}" ]]; then
    echo "[ERROR] Missing ${DATA_DIR}/${raw_file}"
    exit 1
  fi
done

MINIO_ROOT_USER="$(
  kubectl get secret minio-secret -n firefly-platform \
    -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 --decode
)"
MINIO_ROOT_PASSWORD="$(
  kubectl get secret minio-secret -n firefly-platform \
    -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode
)"

echo "[INFO] Seeding raw data files into ${MINIO_BUCKET}/raw via ${MINIO_ENDPOINT}"

UPLOAD_SCRIPT="$(mktemp)"
trap 'rm -f "${UPLOAD_SCRIPT}"' EXIT

{
  printf 'set -e\n'
  printf "mc alias set local '%s' '%s' '%s'\n" \
    "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"
  printf "mc mb -p local/%s || true\n" "${MINIO_BUCKET}"
  for raw_file in "${RAW_FILES[@]}"; do
    printf "mc cp '/seed/%s' 'local/%s/raw/%s'\n" \
      "${raw_file}" "${MINIO_BUCKET}" "${raw_file}"
  done
  printf "mc ls 'local/%s/raw'\n" "${MINIO_BUCKET}"
} > "${UPLOAD_SCRIPT}"

${DOCKER_CMD} run --rm \
  --network host \
  -v "${DATA_DIR}:/seed:ro" \
  -v "${UPLOAD_SCRIPT}:/tmp/upload.sh:ro" \
  --entrypoint /bin/sh \
  minio/mc:latest \
  /tmp/upload.sh

echo "[INFO] Raw data seeded successfully."
