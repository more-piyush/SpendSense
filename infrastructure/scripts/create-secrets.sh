#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f ../k8s/namespace/namespace.yaml

read -rsp "Enter PostgreSQL password: " POSTGRES_PASSWORD
echo
read -rsp "Enter Firefly APP_KEY: " FIREFLY_APP_KEY
echo
read -rp "Enter MinIO root user: " MINIO_ROOT_USER
read -rsp "Enter MinIO root password: " MINIO_ROOT_PASSWORD
echo

kubectl create secret generic postgres-secret   --from-literal=POSTGRES_DB=firefly   --from-literal=POSTGRES_USER=firefly   --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}"   -n firefly-platform   --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic firefly-secret   --from-literal=APP_KEY="${FIREFLY_APP_KEY}"   -n firefly-platform   --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic minio-secret   --from-literal=MINIO_ROOT_USER="${MINIO_ROOT_USER}"   --from-literal=MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}"   -n firefly-platform   --dry-run=client -o yaml | kubectl apply -f -

echo "[INFO] Secrets created."
