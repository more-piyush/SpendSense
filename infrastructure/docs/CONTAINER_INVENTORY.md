# Container Inventory

| Role | Container | Purpose | Manifest Path |
|---|---|---|---|
| DevOps | firefly | Base open-source application | `k8s/firefly/deployment.yaml` |
| DevOps | postgres | App and MLflow database | `k8s/postgres/deployment.yaml` |
| DevOps | minio | Artifact storage | `k8s/minio/deployment.yaml` |
| DevOps | mlflow | Tracking and registry | `k8s/mlflow/deployment.yaml` |
| Serving | inference-dummy | Placeholder serving service | `k8s/inference-dummy/deployment.yaml` |
| Training | training-dummy | Placeholder training job | `k8s/training-dummy/job.yaml` |
