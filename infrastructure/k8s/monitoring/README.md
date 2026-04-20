# SpendSense Monitoring

This directory contains the minimal monitoring setup for the Kubernetes stack.

## What gets installed

- `kube-prometheus-stack` via Helm
- `monitoring` namespace
- a custom `PrometheusRule` for SpendSense deployments, CronJobs, PVCs, and `/data`

## Grafana access

The Helm values expose Grafana as a `NodePort` service on port `30300`.

Default credentials from the values file:

- username: `admin`
- password: `admin123-change-me`

Change the Grafana password in `kube-prometheus-stack-values.yaml` before production use.

## Install flow

```bash
kubectl apply -f infrastructure/k8s/monitoring/namespace.yaml

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f infrastructure/k8s/monitoring/kube-prometheus-stack-values.yaml

kubectl apply -f infrastructure/k8s/monitoring/prometheus-rules.yaml
```

## Useful checks

```bash
kubectl get pods -n monitoring
kubectl get svc -n monitoring
kubectl get prometheusrule -n monitoring
kubectl get servicemonitor -n monitoring
```
