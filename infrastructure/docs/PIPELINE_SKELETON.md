# Pipeline Skeleton

1. Data CronJob builds training datasets and writes them to MinIO (`training-data` bucket).
2. Training CronJobs load feedback + datasets, retrain models, and log runs to MLflow.
3. MLflow persists metadata in PostgreSQL and artifacts in MinIO.
4. Serving deployment exposes online prediction endpoint at NodePort `30081`.
5. Firefly III is exposed at NodePort `30080` and can call serving internally.
6. All external access is via floating IP + NodePort (no domain required).
