#!/usr/bin/env bash
# Local test runner — usage: ./scripts/run_local.sh [categorization|anomaly]
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found. Copy .env.example to .env and fill in credentials." >&2
fi

task="${1:-categorization}"
case "$task" in
    categorization)
        python build_categorization_dataset.py
        ;;
    anomaly|trend|trend_detection)
        python build_trend_detection_dataset.py
        ;;
    both|all)
        python build_categorization_dataset.py
        python build_trend_detection_dataset.py
        ;;
    *)
        echo "Usage: $0 [categorization|anomaly|both]" >&2
        exit 2
        ;;
esac
