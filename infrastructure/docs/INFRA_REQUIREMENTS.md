# Infrastructure Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Persistence |
|---|---:|---:|---:|---:|---|
| Firefly III | 250m | 500m | 512Mi | 1Gi | No |
| PostgreSQL | 250m | 500m | 512Mi | 1Gi | Yes |
| MinIO | 250m | 500m | 256Mi | 512Mi | Yes |
| MLflow | 250m | 500m | 256Mi | 512Mi | No |
| Dummy inference API | 100m | 250m | 128Mi | 256Mi | No |
| Dummy training job | 100m | 250m | 128Mi | 256Mi | No |
