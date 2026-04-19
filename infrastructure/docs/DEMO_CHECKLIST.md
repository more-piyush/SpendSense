# Demo Checklist

## Show these commands
- `kubectl get pods -n firefly-platform`
- `kubectl get svc -n firefly-platform`
- `kubectl get pvc -n firefly-platform`
- `kubectl get cronjobs -n firefly-platform`
- `kubectl logs job/minio-bootstrap -n firefly-platform`

## Show these UIs (Floating IP)
- Firefly: `http://<FLOATING_IP>:30080`
- Serving API health: `http://<FLOATING_IP>:30081/health`
- MLflow: `http://<FLOATING_IP>:30500`
- MinIO Console: `http://<FLOATING_IP>:30901`

## Show persistence
- Delete postgres pod and confirm data remains
- Delete minio pod and confirm PVC remains attached
