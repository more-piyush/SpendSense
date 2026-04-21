#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFRA_DIR}/.." && pwd)"
ENV_FILE="${1:-${INFRA_DIR}/config/deploy.env}"
export DEPLOY_ENV_FILE="${ENV_FILE}"
export SPENDSENSE_ENV_FILE="${ENV_FILE}"
LOG_DIR="${INFRA_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

phase() {
  printf '\n========== %s ==========\n' "$*"
}

info() {
  printf '[INFO] %s\n' "$*"
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

run() {
  printf '[CMD] %s\n' "$*"
  "$@"
}

cleanup_on_error() {
  printf '\n[ERROR] Deployment failed. See %s for the full log.\n' "${LOG_FILE}" >&2
}
trap cleanup_on_error ERR

[[ -f "${ENV_FILE}" ]] || fail "Missing env file: ${ENV_FILE}"
source "${SCRIPT_DIR}/load-env.sh" "${ENV_FILE}"

cd "${REPO_ROOT}"

[[ -n "${FLOATING_IP:-}" ]] || fail "FLOATING_IP must be set in ${ENV_FILE}"

export DEPLOY_USER="${SUDO_USER:-${USER}}"
if [[ "${DEPLOY_USER}" == "root" ]]; then
  export DEPLOY_HOME="/root"
else
  export DEPLOY_HOME="/home/${DEPLOY_USER}"
fi
export DATA_VOLUME_DEVICE="${DATA_VOLUME_DEVICE:-/dev/vdb}"
export DATA_VOLUME_MOUNT_PATH="${DATA_VOLUME_MOUNT_PATH:-/data}"
export DATA_VOLUME_FILESYSTEM="${DATA_VOLUME_FILESYSTEM:-ext4}"
export K3S_INSTALL_CHANNEL="${K3S_INSTALL_CHANNEL:-stable}"
export K3S_SERVER_EXTRA_ARGS="${K3S_SERVER_EXTRA_ARGS:---write-kubeconfig-mode 644 --disable traefik}"
export INSTALL_TERRAFORM="${INSTALL_TERRAFORM:-true}"

phase "Phase 1/9 - OS packages and Ansible"
run sudo apt-get update
run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ansible \
  ca-certificates \
  curl \
  git \
  gnupg \
  jq \
  lsb-release \
  make \
  openssl \
  python3 \
  python3-pip \
  rsync \
  software-properties-common \
  unzip

phase "Phase 2/9 - Base VM bootstrap with Ansible"
run ansible-playbook -i localhost, -c local infrastructure/ansible/single_vm/site.yml

export KUBECONFIG="${DEPLOY_HOME}/.kube/config"
run kubectl get nodes

phase "Phase 3/9 - Build and import runtime images"
images=(
  "firefly-data:latest|Data/pipelines/Dockerfile|Data/pipelines"
  "spendsense/training:latest|Training/training_scripts/Dockerfile|Training/training_scripts"
  "spendsense/serving-unified:latest|serving/Dockerfile.unified|serving"
  "spendsense/firefly-retraining:latest|firefly-retraining/Dockerfile|firefly-retraining"
  "spendsense/firefly-custom:latest|firefly-iii-main/firefly-iii-main/Dockerfile.custom|firefly-iii-main/firefly-iii-main"
)

for spec in "${images[@]}"; do
  IFS='|' read -r tag dockerfile context <<< "${spec}"
  info "Building ${tag}"
  run sudo docker build -t "${tag}" -f "${dockerfile}" "${context}"
  info "Importing ${tag} into K3s containerd"
  run bash -lc "sudo docker save '${tag}' | sudo k3s ctr images import -"
done
run sudo docker builder prune -af

phase "Phase 4/9 - Core namespace, secrets, and platform deployment"
run bash ./infrastructure/scripts/bootstrap.sh
run bash ./infrastructure/scripts/create-secrets-from-env.sh
export FLOATING_IP
run bash ./infrastructure/scripts/deploy-core.sh

phase "Phase 5/9 - Seed raw data and bootstrap one-off data pipeline"
export DOCKER_CMD="sudo docker"
run bash ./infrastructure/scripts/seed-data.sh
run kubectl delete job -n firefly-platform data-pipeline-bootstrap --ignore-not-found
run kubectl create job --from=cronjob/data-pipeline-nightly data-pipeline-bootstrap -n firefly-platform
run kubectl wait --for=condition=complete job/data-pipeline-bootstrap -n firefly-platform --timeout=2h
run kubectl logs -n firefly-platform job/data-pipeline-bootstrap

phase "Phase 6/9 - Initial model training and delayed activation"
run bash ./infrastructure/scripts/run-initial-training.sh

phase "Phase 7/9 - Monitoring stack"
if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
  run bash ./infrastructure/scripts/deploy-monitoring.sh
  run bash ./infrastructure/scripts/validate-monitoring.sh
else
  info "Skipping monitoring because DEPLOY_MONITORING=${DEPLOY_MONITORING:-false}"
fi

phase "Phase 8/9 - Validation and evidence"
run bash ./infrastructure/scripts/validate-core.sh
run bash ./infrastructure/scripts/collect-evidence.sh

phase "Phase 9/9 - Final status"
run kubectl get pods -A -o wide
run kubectl get svc -A
cat <<EOF
[INFO] Deployment completed successfully.
[INFO] Log file: ${LOG_FILE}
[INFO] Firefly III: http://${FLOATING_IP}:30080
[INFO] Serving API: http://${FLOATING_IP}:30081
[INFO] MLflow UI: http://${FLOATING_IP}:30500
[INFO] MinIO API: http://${FLOATING_IP}:30900
[INFO] MinIO Console: http://${FLOATING_IP}:30901
[INFO] Grafana: http://${FLOATING_IP}:30300
EOF
