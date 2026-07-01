# SpendSense

An end-to-end MLOps platform that extends [Firefly III](https://www.firefly-iii.org/)
with ML-driven transaction categorization and spending trend/anomaly detection.
The system covers the full lifecycle — data ingestion, model training, online
serving, canary rollouts, and continuous retraining from user feedback —
deployed on Kubernetes.

## Architecture

```
Data ──► Training ──► Serving ──► Firefly III
  │          │            │
  └────── MLflow / MinIO / PostgreSQL ──────┘
              (tracked by infrastructure/)
```

- **Firefly III** (`firefly-iii-main/`) — the vendored personal-finance manager
  that end users interact with; the serving API augments it with automatic
  categorization and trend/anomaly detection.
- **Data** (`Data/`) — ingestion and data-quality pipelines. Pulls BLS
  Consumer Expenditure survey data, builds synthetic/production datasets, and
  runs Soda data-quality checks.
- **firefly-retraining** (`firefly-retraining/`) — scheduled pipeline that
  merges production logs with external training data into versioned,
  reproducible retraining datasets (weekly categorization / monthly trend).
- **Training** (`Training/`) — training and retraining scripts for the
  categorization model (DistilBERT / baseline LR) and trend-detection models
  (XGBoost, Isolation Forest), with MLflow experiment tracking and model
  promotion/canary gating.
- **serving** (`serving/`) — FastAPI serving stack (`service/`) exposing
  categorization, trend, feedback, and Firefly-integration endpoints; loads
  active models from MLflow via `active_models.json`. Also includes
  standalone ONNX/quantized serving examples used during experimentation.
- **infrastructure** (`infrastructure/`) — Terraform (Chameleon/OpenStack),
  Ansible (K3s bootstrap + ArgoCD), and Kubernetes manifests that deploy the
  entire stack, plus operational scripts for deployment, monitoring, and
  canary rollout management.

## Quick start

The fastest path to a running stack is the single-VM bootstrap:

```bash
cd infrastructure
cp config/deploy.env.example config/deploy.env
# edit config/deploy.env: set FLOATING_IP, DATA_VOLUME_DEVICE, etc.
bash scripts/deploy-all.sh config/deploy.env
```

This provisions Docker/K3s/Helm, builds all service images, deploys the core
platform (Firefly III, PostgreSQL, MinIO, MLflow, serving API), seeds data,
runs initial training, and activates models. See
[infrastructure/README.md](infrastructure/README.md) for the full manual
workflow (Terraform + Ansible), GitHub Actions deployment, and teardown.

Once deployed, with `FLOATING_IP` set:

| Service | URL |
|---|---|
| Firefly III | `http://<FLOATING_IP>:30080` |
| Serving API | `http://<FLOATING_IP>:30081` |
| MLflow UI | `http://<FLOATING_IP>:30500` |
| MinIO Console | `http://<FLOATING_IP>:30901` |
| Grafana | `http://<FLOATING_IP>:30300` |

## Repository layout

```
SpendSense/
├── Data/                  # ingestion + data-quality pipelines
├── Training/              # model training / retraining scripts
├── firefly-retraining/    # versioned retraining-dataset builder
├── serving/               # FastAPI serving stack + ONNX experiments
├── firefly-iii-main/      # vendored Firefly III application
├── infrastructure/        # Terraform, Ansible, Kubernetes, deploy scripts
└── .github/workflows/     # CI + remote deployment / demo-rollout automation
```

Each component has its own README with detailed setup and usage instructions:

- [firefly-retraining/README.md](firefly-retraining/README.md)
- [infrastructure/README.md](infrastructure/README.md)
- [infrastructure/ansible/README.md](infrastructure/ansible/README.md)
- [infrastructure/k8s/monitoring/README.md](infrastructure/k8s/monitoring/README.md)

## CI/CD

- `.github/workflows/main.yml` — `Deploy Infrastructure`: provisions Chameleon
  nodes via Terraform, bootstraps K3s via Ansible, and deploys the full stack
  to a fresh VM (manual `workflow_dispatch`).
- `.github/workflows/demo-rollout.yml` — `Demo Retrain And Canary`: drives
  retraining and canary rollout/promotion/rollback against an already-running
  VM (manual `workflow_dispatch`).
