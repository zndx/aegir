#!/usr/bin/env bash
# Idempotent oneshot start for aegir.service (systemd).
#
# Full-stack doctrine (signals.target): devenv process-compose graph
# (gateway :8091, vite :5173, capability-engine :50151, …). Never use
# foreground `devenv up` — it never returns under a oneshot unit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"
# devenv/nix may need impure + insecure minio for Metaflow local S3
export NIXPKGS_ALLOW_INSECURE="${NIXPKGS_ALLOW_INSECURE:-1}"

GRPC_PORT="${AEGIR_ENGINE_PORT:-50151}"
GATEWAY_URL="${AEGIR_GATEWAY_HEALTH:-http://127.0.0.1:8091/api/health}"
VITE_URL="${AEGIR_VITE_URL:-http://127.0.0.1:5173/}"
POLL_ITERS="${AEGIR_SYSTEMD_POLL_ITERS:-180}"
POLL_SLEEP="${AEGIR_SYSTEMD_POLL_SLEEP:-5}"

info() { echo "aegir.service: $*"; }

status_ok() {
  local py="${ROOT}/.devenv/state/venv/bin/python"
  [[ -x "$py" ]] || py=python3
  AEGIR_ENGINE_PORT="${GRPC_PORT}" "$py" "$ROOT/scripts/zndx_status_ok.py" >/dev/null 2>&1
}

gateway_ok() {
  curl -sf --max-time 3 "${GATEWAY_URL}" >/dev/null 2>&1
}

vite_ok() {
  curl -sf --max-time 3 -o /dev/null "${VITE_URL}" 2>/dev/null
}

stack_ok() {
  status_ok && gateway_ok && vite_ok
}

nlisten() {
  command -v ss >/dev/null 2>&1 || { echo 0; return; }
  ss -ltn 2>/dev/null | grep -cE ":${GRPC_PORT}\\s" || true
}

if [[ "$(nlisten)" -gt 1 ]]; then
  info "WARN: $(nlisten) listeners on :${GRPC_PORT} — full-stack recycle"
elif stack_ok; then
  info "already READY (Status :${GRPC_PORT} + gateway + vite) — skip up"
  exit 0
fi

info "starting full product stack (devenv up -d) incl. capability-engine :${GRPC_PORT}"
# Prefer detached process-compose. Do NOT run bare `devenv up` (blocks forever).
# just up = devenv up -d + stack-health; still fine if it returns.
if ! /bin/bash -lc "cd \"$ROOT\" && export PATH=\"/usr/local/bin:\$PATH\" && export NIXPKGS_ALLOW_INSECURE=1 && (devenv up -d || just up)"; then
  info "up reported failure — will still poll (stack may already be live)"
fi

for i in $(seq 1 "$POLL_ITERS"); do
  if [[ "$(nlisten)" -gt 1 ]]; then
    info "WARN: multi-listener on :${GRPC_PORT} (iter=$i)"
  elif stack_ok; then
    info "full stack ready (Status + gateway + vite) iter=$i"
    # Warm the varnish-fronted waffle roster: the malloc store is empty
    # after a restart. Fire-and-forget; the public route primes the cache.
    (
      sleep 5
      curl -sf --max-time 60 -o /dev/null \
        "http://127.0.0.1:8091/api/aegir/v1/federation/surfaces" || true
    ) >/dev/null 2>&1 &
    exit 0
  fi
  # progress every ~30s
  if (( i % 6 == 0 )); then
    info "waiting… iter=$i status=$(status_ok && echo ok || echo no) gw=$(gateway_ok && echo ok || echo no) vite=$(vite_ok && echo ok || echo no)"
  fi
  sleep "$POLL_SLEEP"
done

info "timed out waiting for full Aegir stack" >&2
info "  Lattice:  scripts/zndx_status_ok.py  (:${GRPC_PORT})" >&2
info "  Gateway:  ${GATEWAY_URL}" >&2
info "  Vite:     ${VITE_URL}" >&2
exit 1
