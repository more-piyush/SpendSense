# SpendSense Monitoring

This directory contains the monitoring stack for the SpendSense platform running
on Kubernetes.

The setup is built around `kube-prometheus-stack` and extends it with:

- blackbox monitoring for service health endpoints
- MinIO metrics scraping
- custom Prometheus alert rules for the platform
- Grafana dashboards tailored to SpendSense workloads

## Components installed

### Core monitoring stack

Installed with Helm from `kube-prometheus-stack`:

- Prometheus
- Alertmanager
- Grafana
- Prometheus Operator
- kube-state-metrics
- node-exporter

These give cluster-level visibility into:

- nodes
- pods
- deployments
- PVCs
- kubelet / kube-state metrics
- filesystem and memory pressure

### Blackbox exporter

Installed from `blackbox-exporter.yaml`.

This exporter probes HTTP endpoints and exposes metrics like:

- `probe_success`
- `probe_duration_seconds`
- `probe_http_status_code`

It is used to blackbox-check service health from inside the cluster.

### MinIO metrics scraping

MinIO is configured with:

- `MINIO_PROMETHEUS_AUTH_TYPE=public`

and scraped via a `ServiceMonitor` at:

- `/minio/v2/metrics/cluster`

## What is monitored

### Cluster and node monitoring

Collected by kube-prometheus-stack:

- node CPU, memory, filesystem, and pressure
- kubelet and pod lifecycle metrics
- deployment availability
- PVC usage
- pod restart counts

### SpendSense service health endpoints

Collected by blackbox exporter via `Probe`:

- `serving-baseline.firefly-platform.svc.cluster.local:8000/health`
- `minio.firefly-platform.svc.cluster.local:9000/minio/health/live`
- `mlflow.firefly-platform.svc.cluster.local:5000/`
- `firefly.firefly-platform.svc.cluster.local:8080/`

These checks tell you if the endpoint is reachable and whether it returns a
successful HTTP response.

### MinIO operational metrics

Collected by the MinIO `ServiceMonitor`.

This gives direct Prometheus scraping of MinIO cluster metrics in addition to
the blackbox liveness probe.

### Kubernetes workload health in `firefly-platform`

The alert rules and dashboards focus on:

- `serving-baseline`
- `mlflow`
- `postgres`
- `minio`
- `firefly`
- CronJobs:
  - `data-pipeline-nightly`
  - `nightly-eval`
  - `monthly-retrain`

### Storage monitoring

This setup monitors:

- `/data` usage on the attached data volume
- PVC usage for:
  - `postgres-pvc`
  - `minio-pvc`
  - `training-state-pvc`

## Alerts created

The custom `PrometheusRule` creates these alerts:

### Deployment availability alerts

- `SpendSenseServingUnavailable`
- `SpendSenseMLflowUnavailable`
- `SpendSensePostgresUnavailable`
- `SpendSenseMinioUnavailable`
- `SpendSenseFireflyUnavailable`

These fire when the corresponding deployment has unavailable replicas for a
sustained period.

### Runtime stability alerts

- `SpendSenseFrequentPodRestarts`

This fires when containers in `firefly-platform` restart repeatedly.

### Scheduled workload alerts

- `SpendSenseCronJobFailures`

This fires when one of the platform CronJobs fails within the lookback window.

### Storage alerts

- `SpendSensePVCUsageHigh`
- `SpendSenseDataDiskUsageHigh`

These watch PVC capacity and the mounted `/data` volume.

### Blackbox probe alerts

- `SpendSenseBlackboxProbeFailed`
- `SpendSenseProbeLatencyHigh`

These fire when health endpoints stop responding successfully or become slow.

### MinIO scrape alert

- `SpendSenseMinioMetricsDown`

This fires when Prometheus cannot scrape MinIO metrics.

## Dashboards included

### 1. SpendSense Overview

Loaded from the Grafana ConfigMap `spendsense-overview-dashboard`.

Panels included:

- Serving probe success
- MinIO probe success
- MLflow probe success
- Firefly probe success
- Pod restarts over time
- PVC usage percentage
- `/data` disk usage percentage
- Blackbox probe duration

This is the main operational dashboard for platform health.

### 2. SpendSense Jobs

Loaded from the Grafana ConfigMap `spendsense-jobs-dashboard`.

Panels included:

- CronJob failures over 24h
- CronJob successes over 24h
- Recent failed jobs table

This is the main dashboard for scheduled pipelines and retraining workflows.

## Files in this directory

- `namespace.yaml`
  Creates the `monitoring` namespace.
- `kube-prometheus-stack-values.yaml`
  Helm values for the monitoring stack.
- `blackbox-exporter.yaml`
  Deploys the blackbox exporter and its self-scrape `ServiceMonitor`.
- `probes.yaml`
  Defines cluster-internal HTTP health probes for key services.
- `minio-servicemonitor.yaml`
  Scrapes MinIO metrics.
- `prometheus-rules.yaml`
  Defines custom SpendSense alert rules.
- `grafana-dashboards.yaml`
  Installs Grafana dashboard ConfigMaps.

## Grafana access

Grafana is exposed as a `NodePort` on port `30300`.

Default credentials:

- username: `admin`
- password: `admin123-change-me`

Change the Grafana password in `kube-prometheus-stack-values.yaml` before
production use.

## Deployment flow

```bash
kubectl apply -f infrastructure/k8s/monitoring/namespace.yaml

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f infrastructure/k8s/monitoring/kube-prometheus-stack-values.yaml

kubectl apply -f infrastructure/k8s/monitoring/blackbox-exporter.yaml
kubectl apply -f infrastructure/k8s/monitoring/probes.yaml
kubectl apply -f infrastructure/k8s/monitoring/minio-servicemonitor.yaml
kubectl apply -f infrastructure/k8s/monitoring/prometheus-rules.yaml
kubectl apply -f infrastructure/k8s/monitoring/grafana-dashboards.yaml
```

## Useful validation commands

```bash
kubectl get pods -n monitoring
kubectl get svc -n monitoring
kubectl get prometheusrule -n monitoring
kubectl get servicemonitor -n monitoring
kubectl get probe -n monitoring
kubectl get configmap -n monitoring | grep dashboard
```

## Notes

- The blackbox probes use in-cluster service DNS names, so they test reachability
  from inside the cluster rather than from the public internet.
- MinIO metrics scraping depends on `MINIO_PROMETHEUS_AUTH_TYPE=public`.
- If you later expose services through Ingress, you may also want a second layer
  of blackbox probes against the public URLs.
