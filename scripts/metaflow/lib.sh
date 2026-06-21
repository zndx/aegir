#!/usr/bin/env bash
# Shared helpers for the aegir Metaflow process scripts (mirrors gaius scripts/lib/process-helpers.sh).
set -euo pipefail

banner() {
  printf '╔══════════════════════════════════════════════════════════════╗\n'
  printf '║  %-59s ║\n' "$1"
  printf '╚══════════════════════════════════════════════════════════════╝\n\n'
}

# Exit 0 if the named env var is "true" (one-shot processes that shouldn't block).
check_disabled_exit() {
  if [ "${!1:-false}" == "true" ]; then echo "$2 disabled ($1=true)"; exit 0; fi
}

# Sleep forever if disabled (long-running processes — keeps process-compose happy).
check_disabled() {
  if [ "${!1:-false}" == "true" ]; then echo "$2 disabled ($1=true)"; sleep infinity; fi
}

wait_for_postgres() {
  local user="${1:-$USER}"
  echo "Waiting for PostgreSQL on :${PGPORT}..."
  for i in $(seq 1 30); do
    if pg_isready -h 127.0.0.1 -p "${PGPORT}" -U "$user" >/dev/null 2>&1; then echo "  ready"; return 0; fi
    [ "$i" -eq 30 ] && { echo "ERROR: postgres not ready after 30s"; exit 1; }
    sleep 1
  done
}

# The aegir coexistence footprint on the shared RKE2 (distinct from gaius's default/9010/30180).
export AEGIR_MF_NAMESPACE="${AEGIR_MF_NAMESPACE:-aegir-metaflow}"
export AEGIR_MINIO_PORT="${AEGIR_MINIO_PORT:-9012}"
export AEGIR_HOST_IP="${AEGIR_HOST_IP:-192.168.1.55}"   # tinybox host IP reachable from pods
export KUBECONFIG="${KUBECONFIG:-$HOME/.config/kube/rke2.yaml}"
