# Infrastructure Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Persistence |
|---|---:|---:|---:|---:|---|
| Firefly III | 100m | 250m | 256Mi | 512Mi | No |
| PostgreSQL | 100m | 250m | 256Mi | 512Mi | Yes |
| MinIO | 100m | 250m | 256Mi | 512Mi | Yes |
| MLflow | 100m | 250m | 256Mi | 512Mi | No |
| Serving baseline API | 500m | 1000m | 1Gi | 2Gi | No |
| Data pipeline CronJob | 250m | 1000m | 512Mi | 2Gi | Ephemeral |
| Training retrain CronJobs | 500m | 2000m | 1Gi | 4Gi | Ephemeral |
