#!/usr/bin/env bash
set -euo pipefail

NS="monitoring"

echo "[INFO] Monitoring namespace resources"
kubectl get pods -n "$NS" -o wide
kubectl get svc -n "$NS"
kubectl get prometheusrule -n "$NS"

echo "[INFO] Rollout status"
kubectl rollout status deployment/monitoring-grafana -n "$NS" --timeout=240s
kubectl rollout status deployment/monitoring-kube-prometheus-operator -n "$NS" --timeout=240s

echo "[INFO] Grafana access"
echo "[INFO] Grafana: http://${FLOATING_IP:-<FLOATING_IP>}:30300"
