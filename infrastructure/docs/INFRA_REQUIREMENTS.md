# Infrastructure Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Persistence |
|---|---:|---:|---:|---:|---|
| Firefly III | 500m | 2 | 1Gi | 4Gi | No |
| PostgreSQL | 500m | 2 | 2Gi | 8Gi | Yes |
| MinIO | 500m | 2 | 2Gi | 8Gi | Yes |
| MLflow | 500m | 2 | 2Gi | 8Gi | No |
| Serving baseline API | 500m | 1000m | 1Gi | 2Gi | No |
| Data pipeline CronJob | 1 | 4 | 2Gi | 8Gi | Ephemeral |
| Training retrain CronJobs | 2 | 8 | 8Gi | 32Gi | Persistent (`training-state-pvc`) |
