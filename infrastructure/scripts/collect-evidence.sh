#!/usr/bin/env bash
set -euo pipefail

OUT="evidence_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

kubectl get nodes -o wide > "$OUT/get_nodes.txt"
kubectl get pods -A -o wide > "$OUT/get_pods.txt"
kubectl get svc -A > "$OUT/get_services.txt"
kubectl get pvc -A > "$OUT/get_pvcs.txt"
kubectl get cronjobs -A > "$OUT/get_cronjobs.txt"

echo "[INFO] Evidence written to $OUT"
