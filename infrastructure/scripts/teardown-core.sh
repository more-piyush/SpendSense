#!/usr/bin/env bash
set -euo pipefail

kubectl delete -f ../k8s/cronjobs/monthly-retrain.yaml --ignore-not-found=true
kubectl delete -f ../k8s/cronjobs/nightly-eval.yaml --ignore-not-found=true
kubectl delete -f ../k8s/training/configmap.yaml --ignore-not-found=true

kubectl delete -f ../k8s/data/cronjob.yaml --ignore-not-found=true
kubectl delete -f ../k8s/data/configmap.yaml --ignore-not-found=true

kubectl delete -f ../k8s/serving/service.yaml --ignore-not-found=true
kubectl delete -f ../k8s/serving/deployment.yaml --ignore-not-found=true

kubectl delete -f ../k8s/firefly/service.yaml --ignore-not-found=true
kubectl delete -f ../k8s/firefly/deployment.yaml --ignore-not-found=true
kubectl delete -f ../k8s/firefly/configmap.yaml --ignore-not-found=true

kubectl delete -f ../k8s/minio-bootstrap/job.yaml --ignore-not-found=true
kubectl delete -f ../k8s/mlflow/service.yaml --ignore-not-found=true
kubectl delete -f ../k8s/mlflow/deployment.yaml --ignore-not-found=true
kubectl delete -f ../k8s/minio/service.yaml --ignore-not-found=true
kubectl delete -f ../k8s/minio/deployment.yaml --ignore-not-found=true
kubectl delete -f ../k8s/postgres/service.yaml --ignore-not-found=true
kubectl delete -f ../k8s/postgres/deployment.yaml --ignore-not-found=true

echo "[INFO] PVCs and namespace preserved by default."
