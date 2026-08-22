#!/usr/bin/env bash
# Peer-scoped unit stop for aegir.service (systemd).
#
# Full stop of THIS project's devenv process-compose graph (UI + gateway +
# capability-engine + workers). Co-tenant-safe: no host teardown / gpu wipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"

GRPC_PORT="${AEGIR_ENGINE_PORT:-50151}"

echo "aegir.service: peer-scoped full-stack stop (devenv down + engine on :${GRPC_PORT})"

/bin/bash -lc "cd \"$ROOT\" && export PATH=\"/usr/local/bin:\$PATH\" && (just down 2>/dev/null || devenv processes down || true)" || true

stop_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local cmd
  cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$cmd" != *aegir.engine* ]]; then
    return 0
  fi
  echo "aegir.service: TERM process group of pid=$pid ($cmd)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

if command -v ss >/dev/null 2>&1; then
  for pid in $(ss -ltnp 2>/dev/null | grep ":${GRPC_PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u); do
    stop_pid "$pid"
  done
fi

sleep 3
if command -v ss >/dev/null 2>&1; then
  for pid in $(ss -ltnp 2>/dev/null | grep ":${GRPC_PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u); do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ "$cmd" == *aegir.engine* ]]; then
      echo "aegir.service: KILL process group of pid=$pid after grace"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${GRPC_PORT}\\s"; then
  echo "aegir.service: WARN :${GRPC_PORT} still listening after stop" >&2
else
  echo "aegir.service: :${GRPC_PORT} free"
fi
