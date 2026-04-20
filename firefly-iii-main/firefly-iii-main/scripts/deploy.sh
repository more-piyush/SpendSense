#!/usr/bin/env bash

set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_APP="${SERVE_APP:-false}"
HOST="${DEPLOY_HOST:-0.0.0.0}"
PORT="${DEPLOY_PORT:-8080}"

log() {
  printf '\n[deploy] %s\n' "$1"
}

fail() {
  printf '\n[deploy][error] %s\n' "$1" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

check_env_file() {
  if [[ ! -f "$APP_ROOT/.env" ]]; then
    fail "Missing $APP_ROOT/.env. Copy .env.example to .env and set your environment variables first."
  fi
}

check_required_env() {
  local env_file="$APP_ROOT/.env"
  if ! grep -Eq '^APP_KEY=' "$env_file"; then
    log "APP_KEY not found in .env. The script will generate one."
  fi
  if ! grep -Eq '^SPENDSENSE_SERVING_URL=.+$' "$env_file"; then
    log "SPENDSENSE_SERVING_URL is empty in .env. Category prediction will be disabled until you set it."
  fi
}

generate_app_key_if_needed() {
  local env_file="$APP_ROOT/.env"
  local current_key
  current_key="$(grep -E '^APP_KEY=' "$env_file" | tail -n 1 | cut -d'=' -f2- || true)"
  if [[ -z "$current_key" || "$current_key" == "SomeRandomStringOf32CharsExactly" ]]; then
    log "Generating Laravel application key"
    (cd "$APP_ROOT" && php artisan key:generate --force)
  fi
}

install_php_dependencies() {
  log "Installing PHP dependencies"
  (cd "$APP_ROOT" && composer install --no-interaction --prefer-dist --optimize-autoloader)
}

install_js_dependencies() {
  log "Installing Node workspace dependencies"
  (cd "$APP_ROOT" && npm install)
}

build_assets() {
  log "Building legacy v1 assets"
  (cd "$APP_ROOT" && npm --workspace resources/assets/v1 run production)

  log "Building v2 assets"
  (cd "$APP_ROOT" && npm --workspace resources/assets/v2 run build)
}

prepare_laravel() {
  log "Running Laravel setup commands"
  (
    cd "$APP_ROOT"
    php artisan migrate --force
    php artisan firefly-iii:upgrade-database
    php artisan firefly-iii:laravel-passport-keys
    php artisan storage:link || true
    php artisan config:clear
    php artisan route:clear
    php artisan view:clear
    php artisan cache:clear
  )
}

show_summary() {
  cat <<EOF

[deploy] Firefly III deployment is complete.

App root: $APP_ROOT
Serving URL from .env: $(grep -E '^SPENDSENSE_SERVING_URL=' "$APP_ROOT/.env" | tail -n 1 | cut -d'=' -f2- || true)

Next:
  1. Verify the app loads in the browser.
  2. Open "Create transaction" and confirm the category suggestion appears.
EOF
}

serve_if_requested() {
  if [[ "$SERVE_APP" == "true" ]]; then
    log "Starting Firefly III on http://$HOST:$PORT"
    cd "$APP_ROOT"
    exec php artisan serve --host="$HOST" --port="$PORT"
  fi
}

main() {
  need_cmd composer
  need_cmd npm
  need_cmd php

  check_env_file
  check_required_env
  install_php_dependencies
  generate_app_key_if_needed
  install_js_dependencies
  build_assets
  prepare_laravel
  show_summary
  serve_if_requested
}

main "$@"
