#!/usr/bin/env bash
set -euo pipefail

NS="firefly-platform"

echo "[INFO] Namespaces / pods / svc / pvc"
kubectl get ns
kubectl get pods -n "$NS" -o wide
kubectl get svc -n "$NS"
kubectl get pvc -n "$NS"

echo "[INFO] Rollout status"
kubectl rollout status deployment/postgres -n "$NS" --timeout=240s
kubectl rollout status deployment/minio -n "$NS" --timeout=240s
kubectl rollout status deployment/mlflow -n "$NS" --timeout=240s
kubectl rollout status deployment/firefly -n "$NS" --timeout=240s
kubectl rollout status deployment/inference-dummy -n "$NS" --timeout=240s

echo "[INFO] Training job logs"
kubectl logs job/training-dummy -n "$NS" || true

echo "[INFO] Firefly URL: http://<NODE_IP>:30080"
echo "[INFO] MLflow: kubectl port-forward svc/mlflow 5000:5000 -n $NS"
echo "[INFO] MinIO console: kubectl port-forward svc/minio 9001:9001 -n $NS"
