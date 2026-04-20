#!/usr/bin/env bash
set -euo pipefail

NS="firefly-platform"

echo "[INFO] Namespaces / pods / svc / pvc / cronjobs"
kubectl get ns
kubectl get pods -n "$NS" -o wide
kubectl get svc -n "$NS"
kubectl get pvc -n "$NS"
kubectl get cronjobs -n "$NS"

echo "[INFO] Rollout status"
kubectl rollout status deployment/postgres -n "$NS" --timeout=240s
kubectl rollout status deployment/minio -n "$NS" --timeout=240s
kubectl rollout status deployment/mlflow -n "$NS" --timeout=240s
kubectl rollout status deployment/firefly -n "$NS" --timeout=240s
kubectl rollout status deployment/serving-baseline -n "$NS" --timeout=240s

echo "[INFO] MinIO bootstrap logs"
kubectl logs job/minio-bootstrap -n "$NS" || true
echo "[INFO] PostgreSQL bootstrap logs"
kubectl logs job/postgres-bootstrap -n "$NS" || true
echo "[INFO] Firefly bootstrap logs"
kubectl logs job/firefly-bootstrap -n "$NS" || true

echo "[INFO] Access URLs"
echo "[INFO] Firefly: http://${FLOATING_IP:-<FLOATING_IP>}:30080"
echo "[INFO] Serving API: http://${FLOATING_IP:-<FLOATING_IP>}:30081"
echo "[INFO] MLflow: http://${FLOATING_IP:-<FLOATING_IP>}:30500"
echo "[INFO] MinIO Console: http://${FLOATING_IP:-<FLOATING_IP>}:30901"
