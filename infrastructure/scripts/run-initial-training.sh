#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${REPO_ROOT}/infrastructure/k8s"
NS="firefly-platform"

WARMUP_SECONDS="${INITIAL_TRAINING_WARMUP_SECONDS:-600}"
RETRY_SECONDS="${ACTIVE_MODEL_RETRY_SECONDS:-120}"
MAX_WAIT_SECONDS="${ACTIVE_MODEL_MAX_WAIT_SECONDS:-600}"

job_manifests=(
  "train-cat-logreg-baseline.yaml"
  "train-cat-distilbert-base.yaml"
  "train-cat-distilbert-v1.yaml"
  "train-cat-distilbert-v2.yaml"
  "train-trend-rf-baseline.yaml"
  "train-trend-xgb-v1.yaml"
  "train-trend-xgb-optuna.yaml"
)

job_names=(
  "train-cat-logreg-baseline"
  "train-cat-distilbert-base"
  "train-cat-distilbert-v1"
  "train-cat-distilbert-v2"
  "train-trend-rf-baseline"
  "train-trend-xgb-v1"
  "train-trend-xgb-optuna"
)

log() {
  printf '[TRAINING] %s\n' "$*"
}

show_status() {
  kubectl get jobs,pods -n "${NS}" | awk 'NR==1 || /train-|set-active-models/'
}

attempt_set_active() {
  kubectl delete job -n "${NS}" set-active-models --ignore-not-found >/dev/null 2>&1 || true
  kubectl apply -f "${K8S_DIR}/training-jobs/set-active-models.yaml"
  if kubectl wait --for=condition=complete -n "${NS}" job/set-active-models --timeout=20m; then
    kubectl rollout restart deployment/serving-baseline -n "${NS}"
    kubectl rollout status deployment/serving-baseline -n "${NS}" --timeout=20m
    return 0
  fi
  kubectl logs -n "${NS}" job/set-active-models || true
  return 1
}

log "Applying training configuration."
kubectl apply -f "${K8S_DIR}/training/candidate-configmap.yaml"
kubectl apply -f "${K8S_DIR}/training/configmap.yaml"
kubectl apply -f "${K8S_DIR}/retraining/configmap.yaml"

log "Deleting any previous initial-training jobs."
kubectl delete job -n "${NS}" --ignore-not-found "${job_names[@]}" >/dev/null 2>&1 || true

log "Launching all initial training jobs in parallel."
for manifest in "${job_manifests[@]}"; do
  kubectl apply -f "${K8S_DIR}/training-jobs/${manifest}"
done
show_status

remaining="${WARMUP_SECONDS}"
while (( remaining > 0 )); do
  sleep_for=60
  if (( remaining < sleep_for )); then
    sleep_for="${remaining}"
  fi
  sleep "${sleep_for}"
  remaining=$((remaining - sleep_for))
  elapsed=$((WARMUP_SECONDS - remaining))
  log "Warmup elapsed: ${elapsed}s/${WARMUP_SECONDS}s"
  show_status
done

log "Warmup complete. Attempting first active-model selection."
if attempt_set_active; then
  log "Active models selected and serving restarted."
  exit 0
fi

log "Initial activation attempt did not succeed yet. Retrying until models are ready."
elapsed_wait=0
while (( elapsed_wait < MAX_WAIT_SECONDS )); do
  sleep "${RETRY_SECONDS}"
  elapsed_wait=$((elapsed_wait + RETRY_SECONDS))
  log "Retrying active-model selection after ${elapsed_wait}s of additional wait."
  show_status
  if attempt_set_active; then
    log "Active models selected and serving restarted."
    exit 0
  fi
done

log "Failed to activate models within ${MAX_WAIT_SECONDS}s after warmup."
exit 1
