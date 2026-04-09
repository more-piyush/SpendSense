#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f ../k8s/namespace/namespace.yaml
kubectl apply -f ../k8s/storage/pvc-postgres.yaml
kubectl apply -f ../k8s/storage/pvc-minio.yaml

kubectl apply -f ../k8s/postgres/deployment.yaml
kubectl apply -f ../k8s/postgres/service.yaml

kubectl apply -f ../k8s/minio/deployment.yaml
kubectl apply -f ../k8s/minio/service.yaml

kubectl apply -f ../k8s/mlflow/deployment.yaml
kubectl apply -f ../k8s/mlflow/service.yaml

kubectl apply -f ../k8s/firefly/configmap.yaml
kubectl apply -f ../k8s/firefly/deployment.yaml
kubectl apply -f ../k8s/firefly/service.yaml

kubectl apply -f ../k8s/inference-dummy/deployment.yaml
kubectl apply -f ../k8s/inference-dummy/service.yaml

kubectl apply -f ../k8s/training-dummy/job.yaml
kubectl apply -f ../k8s/cronjobs/nightly-eval.yaml
kubectl apply -f ../k8s/cronjobs/monthly-retrain.yaml

echo "[INFO] Core services deployed."
