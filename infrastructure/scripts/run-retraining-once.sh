#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"

NS="firefly-platform"
TRAINING_IMAGE="${TRAINING_IMAGE:-spendsense/training:latest}"

TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <categorization|trend> [--force]" >&2
  exit 1
fi
shift || true

FORCE_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "${TASK}" in
  categorization)
    JOB_PREFIX="demo-categorization-retrain"
    CONFIG_SUBPATH="retrain_categorization.yaml"
    ;;
  trend)
    JOB_PREFIX="demo-trend-retrain"
    CONFIG_SUBPATH="retrain_trend.yaml"
    ;;
  *)
    echo "Unsupported task: ${TASK}" >&2
    exit 1
    ;;
esac

JOB_NAME="${JOB_PREFIX}-$(date +%Y%m%d%H%M%S)"
echo "[RETRAIN] Launching ${JOB_NAME}"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NS}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: ${JOB_PREFIX}
          image: ${TRAINING_IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - python
            - retrain.py
            - /app/runtime-config/${CONFIG_SUBPATH}
$(if [[ -n "${FORCE_FLAG}" ]]; then cat <<'INNER'
            - --force
INNER
fi)
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: POSTGRES_PASSWORD
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-secret
                  key: MINIO_ROOT_USER
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-secret
                  key: MINIO_ROOT_PASSWORD
            - name: MLFLOW_S3_ENDPOINT_URL
              value: http://minio.firefly-platform.svc.cluster.local:9000
          volumeMounts:
            - name: runtime-config
              mountPath: /app/runtime-config/${CONFIG_SUBPATH}
              subPath: ${CONFIG_SUBPATH}
            - name: training-data
              mountPath: /data
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
            limits:
              cpu: "8"
              memory: "32Gi"
      volumes:
        - name: runtime-config
          configMap:
            name: training-config
        - name: training-data
          persistentVolumeClaim:
            claimName: training-state-pvc
EOF

kubectl wait --for=condition=complete -n "${NS}" "job/${JOB_NAME}" --timeout=12h || {
  kubectl logs -n "${NS}" "job/${JOB_NAME}" || true
  exit 1
}
kubectl logs -n "${NS}" "job/${JOB_NAME}"
