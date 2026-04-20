#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../k8s/monitoring"

kubectl apply -f "${K8S_DIR}/namespace.yaml"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo update >/dev/null

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f "${K8S_DIR}/kube-prometheus-stack-values.yaml"

kubectl apply -f "${K8S_DIR}/prometheus-rules.yaml"

echo "[INFO] Monitoring stack deployed."
