#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/infrastructure/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/remote_deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

phase() {
  printf '\n========== %s ==========\n' "$*"
}

run() {
  printf '[CMD] %s\n' "$*"
  "$@"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[ERROR] Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

cleanup_on_error() {
  printf '\n[ERROR] Remote deployment failed. See %s for details.\n' "${LOG_FILE}" >&2
}
trap cleanup_on_error ERR

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
export DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
export BUILD_CACHE_ROOT="${BUILD_CACHE_ROOT:-/data/tmp}"
export TMPDIR="${BUILD_CACHE_ROOT}"

if [[ ! -f "${KUBECONFIG}" && -f /etc/rancher/k3s/k3s.yaml ]]; then
  mkdir -p "${HOME}/.kube"
  sudo cp /etc/rancher/k3s/k3s.yaml "${KUBECONFIG}"
  sudo chown "$(id -u)":"$(id -g)" "${KUBECONFIG}"
  chmod 600 "${KUBECONFIG}"
fi

phase_prerequisites() {
  phase "Control-plane prerequisites"
  require_cmd docker
  require_cmd jq
  require_cmd rsync
  require_cmd unzip
  require_cmd helm
  run sudo systemctl enable --now docker
  run sudo mkdir -p "${BUILD_CACHE_ROOT}"
  run sudo chown "$(id -u):$(id -g)" "${BUILD_CACHE_ROOT}"
  run sudo chmod 1777 "${BUILD_CACHE_ROOT}"
}

phase_kubernetes_access() {
  phase "Kubernetes access"
  run kubectl get nodes
}

phase_build_images() {
  phase "Build and import runtime images"
  cd "${REPO_ROOT}"
  run mkdir -p "${BUILD_CACHE_ROOT}"
  images=(
    "firefly-data:latest|Data/pipelines/Dockerfile|Data/pipelines"
    "spendsense/training:latest|Training/training_scripts/Dockerfile|Training/training_scripts"
    "spendsense/serving-unified:latest|serving/Dockerfile.unified|serving"
    "spendsense/firefly-retraining:latest|firefly-retraining/Dockerfile|firefly-retraining"
    "spendsense/firefly-custom:latest|firefly-iii-main/firefly-iii-main/Dockerfile.custom|firefly-iii-main/firefly-iii-main"
  )

  for spec in "${images[@]}"; do
    IFS='|' read -r tag dockerfile context <<< "${spec}"
    run sudo docker build -t "${tag}" -f "${dockerfile}" "${context}"
    run bash -lc "sudo docker save '${tag}' | sudo k3s ctr images import -"
  done
  run sudo docker builder prune -af
}

phase_deploy_core() {
  phase "Secrets and core platform"
  run bash infrastructure/scripts/bootstrap.sh
  run bash infrastructure/scripts/create-secrets-from-env.sh
  run bash infrastructure/scripts/deploy-core.sh
}

phase_seed_data() {
  phase "Data bootstrap"
  run bash infrastructure/scripts/seed-data.sh
  run kubectl delete job -n firefly-platform data-pipeline-bootstrap --ignore-not-found
  run kubectl create job --from=cronjob/data-pipeline-nightly data-pipeline-bootstrap -n firefly-platform
  run kubectl wait --for=condition=complete job/data-pipeline-bootstrap -n firefly-platform --timeout=2h
  run kubectl logs -n firefly-platform job/data-pipeline-bootstrap
}

phase_train_models() {
  phase "Initial model training"
  run bash infrastructure/scripts/run-initial-training.sh
}

phase_monitoring() {
  phase "Monitoring"
  if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
    run bash infrastructure/scripts/deploy-monitoring.sh
    run bash infrastructure/scripts/validate-monitoring.sh
  else
    printf '[INFO] Monitoring deployment skipped.\n'
  fi
}

phase_validate() {
  phase "Validation"
  run bash infrastructure/scripts/validate-core.sh
  run bash infrastructure/scripts/collect-evidence.sh
  run kubectl get pods -A -o wide
  run kubectl get svc -A
}

print_summary() {
  cat <<EOF
[INFO] Remote deployment completed successfully.
[INFO] Log file: ${LOG_FILE}
[INFO] Firefly III: http://${FLOATING_IP}:30080
[INFO] Serving API: http://${FLOATING_IP}:30081
[INFO] MLflow UI: http://${FLOATING_IP}:30500
[INFO] MinIO API: http://${FLOATING_IP}:30900
[INFO] MinIO Console: http://${FLOATING_IP}:30901
[INFO] Grafana: http://${FLOATING_IP}:30300
EOF
}

step="${1:-all}"

case "${step}" in
  prerequisites)
    phase_prerequisites
    ;;
  kubernetes-access)
    phase_kubernetes_access
    ;;
  build-images)
    phase_build_images
    ;;
  deploy-core)
    phase_deploy_core
    ;;
  seed-data)
    phase_seed_data
    ;;
  train-models)
    phase_train_models
    ;;
  monitoring)
    phase_monitoring
    ;;
  validate)
    phase_validate
    print_summary
    ;;
  all)
    phase_prerequisites
    phase_kubernetes_access
    phase_build_images
    phase_deploy_core
    phase_seed_data
    phase_train_models
    phase_monitoring
    phase_validate
    print_summary
    ;;
  *)
    printf 'Usage: %s [all|prerequisites|kubernetes-access|build-images|deploy-core|seed-data|train-models|monitoring|validate]\n' "$0" >&2
    exit 1
    ;;
esac
