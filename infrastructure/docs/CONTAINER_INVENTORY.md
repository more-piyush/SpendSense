# Container Inventory

| Role | Container | Purpose | Manifest Path |
|---|---|---|---|
| DevOps | firefly | Firefly III application | `k8s/firefly/deployment.yaml` |
| DevOps | postgres | App + MLflow backend database | `k8s/postgres/deployment.yaml` |
| DevOps | minio | Artifact + dataset object storage | `k8s/minio/deployment.yaml` |
| DevOps | mlflow | Tracking and model registry UI/API | `k8s/mlflow/deployment.yaml` |
| Serving | serving-baseline | Online inference API | `k8s/serving/deployment.yaml` |
| Data | spendsense/data-pipeline | Nightly ETL to training datasets | `k8s/data/cronjob.yaml` |
| Training | spendsense/training | Scheduled model retraining | `k8s/cronjobs/nightly-eval.yaml` / `k8s/cronjobs/monthly-retrain.yaml` |
| Platform | minio/mc | Bucket bootstrap init job | `k8s/minio-bootstrap/job.yaml` |
