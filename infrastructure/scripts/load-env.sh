#!/usr/bin/env bash

if [[ -n "${SPENDSENSE_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_load_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_load_env_infra_dir="$(cd "${_load_env_script_dir}/.." && pwd)"
_load_env_default_file="${_load_env_infra_dir}/config/deploy.env"
_load_env_target="${SPENDSENSE_ENV_FILE:-${DEPLOY_ENV_FILE:-${1:-${_load_env_default_file}}}}"

if [[ -f "${_load_env_target}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_load_env_target}"
  set +a
fi

export SPENDSENSE_ENV_LOADED=1

