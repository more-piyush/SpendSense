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

phase "Phase 1/7 - Control-plane tools"
run sudo apt-get update
run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  docker.io \
  curl \
  jq \
  rsync \
  unzip
run sudo systemctl enable --now docker
run sudo mkdir -p "${BUILD_CACHE_ROOT}"
run sudo chown "$(id -u):$(id -g)" "${BUILD_CACHE_ROOT}"
run sudo chmod 1777 "${BUILD_CACHE_ROOT}"

if ! command -v helm >/dev/null 2>&1; then
  run bash -lc "curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
fi

phase "Phase 2/7 - Kubernetes access"
run kubectl get nodes

phase "Phase 3/7 - Build and import runtime images"
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

phase "Phase 4/7 - Secrets and core platform"
run bash infrastructure/scripts/bootstrap.sh
run bash infrastructure/scripts/create-secrets-from-env.sh
run bash infrastructure/scripts/deploy-core.sh

phase "Phase 5/7 - Data bootstrap and initial training"
run bash infrastructure/scripts/seed-data.sh
run kubectl delete job -n firefly-platform data-pipeline-bootstrap --ignore-not-found
run kubectl create job --from=cronjob/data-pipeline-nightly data-pipeline-bootstrap -n firefly-platform
run kubectl wait --for=condition=complete job/data-pipeline-bootstrap -n firefly-platform --timeout=2h
run kubectl logs -n firefly-platform job/data-pipeline-bootstrap
run bash infrastructure/scripts/run-initial-training.sh

phase "Phase 6/7 - Monitoring"
if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
  run bash infrastructure/scripts/deploy-monitoring.sh
  run bash infrastructure/scripts/validate-monitoring.sh
fi

phase "Phase 7/7 - Validation"
run bash infrastructure/scripts/validate-core.sh
run bash infrastructure/scripts/collect-evidence.sh
run kubectl get pods -A -o wide
run kubectl get svc -A

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
