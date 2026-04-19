# Demo Checklist

## Show these commands
- `kubectl get pods -n firefly-platform`
- `kubectl get svc -n firefly-platform`
- `kubectl get pvc -n firefly-platform`
- `kubectl logs job/training-dummy -n firefly-platform`

## Show these UIs
- Firefly in browser
- MLflow via port-forward
- MinIO console via port-forward

## Show persistence
- Delete postgres pod and confirm data remains
- Delete minio pod and confirm PVC remains attached
