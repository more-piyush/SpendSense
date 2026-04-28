#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load-env.sh"

NS="${NS:-firefly-platform}"
ROLE_NAME="${POSTGRES_USER:-firefly}"
ROLE_PASSWORD="${POSTGRES_PASSWORD:-}"

if [[ -z "${ROLE_PASSWORD}" ]]; then
  echo "[INFO] PostgreSQL password recovery skipped because POSTGRES_PASSWORD is empty."
  exit 0
fi

if ! kubectl get deployment postgres -n "${NS}" >/dev/null 2>&1; then
  echo "[INFO] PostgreSQL password recovery skipped because deployment/postgres does not exist yet."
  exit 0
fi

kubectl rollout status deployment/postgres -n "${NS}" --timeout=240s >/dev/null

pod_name="$(
  kubectl get pods -n "${NS}" -l app=postgres \
    -o jsonpath='{.items[0].metadata.name}'
)"

if [[ -z "${pod_name}" ]]; then
  echo "[INFO] PostgreSQL password recovery skipped because no postgres pod is ready."
  exit 0
fi

echo "[INFO] Reconciling PostgreSQL password for role ${ROLE_NAME} on pod ${pod_name}."

kubectl exec -i -n "${NS}" "${pod_name}" -- \
  env ROLE_NAME="${ROLE_NAME}" ROLE_PASSWORD="${ROLE_PASSWORD}" bash -lc '
    set -euo pipefail
    cat <<'"'"'SQL'"'"' | su postgres -c "psql -v ON_ERROR_STOP=1 -U \"$ROLE_NAME\" -d postgres -v role_name=\"$ROLE_NAME\" -v role_password=\"$ROLE_PASSWORD\" -f -"
SELECT format('ALTER ROLE %I WITH PASSWORD %L', :'role_name', :'role_password') \gexec
SQL
  '

echo "[INFO] PostgreSQL password recovery completed."
