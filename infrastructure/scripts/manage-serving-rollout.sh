#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${REPO_ROOT}/infrastructure/k8s"
NS="firefly-platform"
REGISTRY_PATH="${REGISTRY_PATH:-s3://mlflow/registry}"
TRAINING_IMAGE="${TRAINING_IMAGE:-spendsense/training:latest}"
PROD_FILE="active_models.json"
CANARY_FILE="active_models_canary.json"

usage() {
  cat <<EOF
Usage:
  $0 prepare [--auto-select] [--categorization-registry-id ID] [--trend-registry-id ID]
  $0 stage <5|10|25>
  $0 promote
  $0 rollback
  $0 status
EOF
}

log() {
  printf '[ROLLOUT] %s\n' "$*"
}

apply_manifests() {
  kubectl apply -f "${K8S_DIR}/serving/deployment.yaml"
  kubectl apply -f "${K8S_DIR}/serving/service.yaml"
  kubectl apply -f "${K8S_DIR}/serving/canary-deployment.yaml"
  kubectl apply -f "${K8S_DIR}/serving/canary-service.yaml"
}

wait_rollouts() {
  kubectl rollout status deployment/serving-baseline -n "${NS}" --timeout=20m
  kubectl rollout status deployment/serving-canary -n "${NS}" --timeout=20m
}

run_training_job() {
  local job_name="$1"
  local command_text="$2"

  kubectl delete job -n "${NS}" "${job_name}" --ignore-not-found=true >/dev/null 2>&1 || true
  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${NS}
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: ${job_name}
          image: ${TRAINING_IMAGE}
          imagePullPolicy: IfNotPresent
          env:
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
            - name: SERVING_ARTIFACT_ROOT
              value: s3://mlflow/serving-artifacts
          command: ["/bin/sh", "-lc"]
          args:
            - |
              set -euo pipefail
              ${command_text}
          volumeMounts:
            - name: training-data
              mountPath: /data
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: training-state-pvc
EOF
  kubectl wait --for=condition=complete -n "${NS}" "job/${job_name}" --timeout=20m
  kubectl logs -n "${NS}" "job/${job_name}"
}

verify_canary() {
  kubectl run canary-curl \
    --rm -i \
    -n "${NS}" \
    --restart=Never \
    --image=curlimages/curl \
    --command -- \
    sh -lc 'curl -fsS http://serving-canary:8000/health && echo && curl -fsS http://serving-canary:8000/ready && echo'
}

prepare_canary() {
  local auto_select="false"
  local cat_registry_id=""
  local trend_registry_id=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --auto-select)
        auto_select="true"
        shift
        ;;
      --categorization-registry-id)
        cat_registry_id="$2"
        shift 2
        ;;
      --trend-registry-id)
        trend_registry_id="$2"
        shift 2
        ;;
      *)
        printf 'Unknown option: %s\n' "$1" >&2
        exit 1
        ;;
    esac
  done

  apply_manifests

  local cmd="python sync_active_models.py --registry-path ${REGISTRY_PATH} --source-file ${PROD_FILE} --target-file ${CANARY_FILE}"
  if [[ "${auto_select}" == "true" ]]; then
    cmd="${cmd} && python set_active_models.py --registry-path ${REGISTRY_PATH} --active-models-file ${CANARY_FILE} --auto-select"
  elif [[ -n "${cat_registry_id}" || -n "${trend_registry_id}" ]]; then
    cmd="${cmd} && python set_active_models.py --registry-path ${REGISTRY_PATH} --active-models-file ${CANARY_FILE}"
    if [[ -n "${cat_registry_id}" ]]; then
      cmd="${cmd} --categorization-registry-id ${cat_registry_id}"
    fi
    if [[ -n "${trend_registry_id}" ]]; then
      cmd="${cmd} --trend-registry-id ${trend_registry_id}"
    fi
  fi

  run_training_job "prepare-serving-canary" "${cmd}"
  kubectl scale deployment/serving-canary -n "${NS}" --replicas=0
  kubectl rollout restart deployment/serving-canary -n "${NS}"
  wait_rollouts
  log "Canary model file prepared in ${CANARY_FILE}."
}

stage_canary() {
  local stage="${1:-}"
  local prod_replicas canary_replicas observation
  case "${stage}" in
    5)
      prod_replicas=19
      canary_replicas=1
      observation="30 minutes"
      ;;
    10)
      prod_replicas=9
      canary_replicas=1
      observation="60 minutes"
      ;;
    25)
      prod_replicas=3
      canary_replicas=1
      observation="2-4 hours"
      ;;
    *)
      printf 'Unsupported canary stage: %s\n' "${stage}" >&2
      exit 1
      ;;
  esac

  apply_manifests
  kubectl scale deployment/serving-baseline -n "${NS}" --replicas="${prod_replicas}"
  kubectl scale deployment/serving-canary -n "${NS}" --replicas="${canary_replicas}"
  wait_rollouts
  verify_canary
  log "Canary stage ${stage}% applied using ${canary_replicas} canary replica(s) and ${prod_replicas} production replica(s)."
  log "Recommended observation window: ${observation}."
}

promote_canary() {
  run_training_job \
    "promote-serving-canary" \
    "python sync_active_models.py --registry-path ${REGISTRY_PATH} --source-file ${CANARY_FILE} --target-file ${PROD_FILE}"
  kubectl scale deployment/serving-canary -n "${NS}" --replicas=0
  kubectl scale deployment/serving-baseline -n "${NS}" --replicas=1
  kubectl rollout restart deployment/serving-baseline -n "${NS}"
  wait_rollouts
  log "Canary promoted to production."
}

rollback_canary() {
  kubectl scale deployment/serving-canary -n "${NS}" --replicas=0
  kubectl scale deployment/serving-baseline -n "${NS}" --replicas=1
  wait_rollouts
  log "Canary traffic removed. Production remains on ${PROD_FILE}."
}

status_rollout() {
  kubectl get deployment -n "${NS}" serving-baseline serving-canary
  kubectl get svc -n "${NS}" serving-baseline serving-canary
}

action="${1:-}"
case "${action}" in
  prepare)
    shift
    prepare_canary "$@"
    ;;
  stage)
    shift
    stage_canary "${1:-}"
    ;;
  promote)
    promote_canary
    ;;
  rollback)
    rollback_canary
    ;;
  status)
    status_rollout
    ;;
  *)
    usage
    exit 1
    ;;
esac
