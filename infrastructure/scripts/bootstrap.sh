#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Verifying kubectl access..."
kubectl version --client
kubectl get nodes
echo "[INFO] Kubernetes access looks good."
