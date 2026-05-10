#!/bin/bash
# Aegir application orchestrator for CAI deployment.
#
# Starts:
#   1. PGlite (if ``AEGIR_DB_URL`` unset and scripts/pglite-server.mjs present)
#   2. Qdrant (if ``qdrant/qdrant`` binary vendored — optional in M1)
#   3. Config resolution (build/config/aegir.env)
#   4. Database bootstrap (schema_migrations + initial schema)
#   5. Gateway (FastAPI, foreground — last process in the chain)
#
# Ported from ``~/local/src/zndx/atelier/bin/start-app.sh`` with M1
# simplifications: no gRPC server, no LLM-provider preflight, no keystone
# agent seeding. Aegir M1 serves only the read-only leaderboard.

set -eo pipefail

cleanup() { pkill -P $$ 2>/dev/null || true; }

for sig in INT QUIT HUP TERM; do
  trap "
    cleanup
    trap - $sig EXIT
    kill -s $sig "'"$$"' "$sig"
done
trap cleanup EXIT

# CAI sets CDSW_APP_PORT; local testing can pass $1 or fall back to 8091.
PORT=${1:-${CDSW_APP_PORT:-8091}}
# CAI binds to 127.0.0.1 (its ingress handles external exposure);
# local testing binds to 0.0.0.0 so ``curl localhost:$PORT`` works
# across containers.
if [ -n "$CDSW_APP_PORT" ]; then
  HOST="127.0.0.1"
else
  HOST="0.0.0.0"
fi
export AEGIR_GATEWAY_HOST="$HOST"
export AEGIR_GATEWAY_PORT="$PORT"

# ── Kill stale processes from previous crash loops ────────────

kill_stale_processes() {
  echo "Cleaning up stale processes..."
  pkill -f "pglite-server.mjs" 2>/dev/null || true
  pkill -f "qdrant/qdrant"     2>/dev/null || true
  pkill -f "aegir.gateway"     2>/dev/null || true
  sleep 1
}
kill_stale_processes

# ── Service readiness helpers ────────────────────────────────

wait_for_service() {
  local name="$1" check_cmd="$2" timeout="${3:-30}" interval="${4:-2}"
  local deadline=$((SECONDS + timeout)) last_err=""
  echo "Waiting for $name..."
  while [ $SECONDS -lt $deadline ]; do
    if last_err=$(eval "$check_cmd" 2>&1); then
      echo "$name ready"
      return 0
    fi
    sleep "$interval"
  done
  echo "ERROR: $name not healthy after ${timeout}s" >&2
  [ -n "$last_err" ] && echo "  last error: $last_err" >&2
  return 1
}

# Proof-of-progress readiness probe for PGlite — extends the deadline
# as long as PGlite is producing observable work (log lines, new files
# in pgdata, socket opening), and fails if nothing progresses for
# STALL_TIMEOUT seconds. See the atelier version for the full rationale.
wait_for_pglite() {
  local port="$1" stall_timeout="${2:-10}" max_timeout="${3:-120}"
  local log_file=".app/pglite.log"
  local deadline=$((SECONDS + max_timeout))
  local last_progress=$SECONDS
  local last_log_lines=0
  local last_file_count=0

  echo "Waiting for PGlite (proof-of-progress, stall=${stall_timeout}s, max=${max_timeout}s)..."

  while [ $SECONDS -lt $deadline ]; do
    local progressed=false

    if [ -f "$log_file" ]; then
      local current_lines
      current_lines=$(wc -l < "$log_file" 2>/dev/null || echo 0)
      if [ "$current_lines" -gt "$last_log_lines" ]; then
        tail -n $((current_lines - last_log_lines)) "$log_file" | while IFS= read -r line; do
          echo "  [pglite] $line"
        done
        last_log_lines=$current_lines
        progressed=true
      fi
    fi

    if [ -d ".app/pgdata" ]; then
      local current_files
      current_files=$(find .app/pgdata -type f 2>/dev/null | wc -l)
      if [ "$current_files" -gt "$last_file_count" ]; then
        echo "  [progress] pgdata: $current_files files (was $last_file_count)"
        last_file_count=$current_files
        progressed=true
      fi
    fi

    if python -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', $port), timeout=2)
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
      if python -c "
from sqlalchemy import create_engine, text
e = create_engine(
    'postgresql+psycopg://postgres:postgres@127.0.0.1:$port/postgres?sslmode=disable&gssencmode=disable',
    connect_args={'connect_timeout': 3},
)
with e.connect() as c:
    c.execute(text('SELECT 1'))
e.dispose()
" 2>/dev/null; then
        echo "PGlite ready (SQL verified)"
        return 0
      else
        echo "  [progress] port open, SQL not yet ready"
        progressed=true
      fi
    fi

    if [ "$progressed" = true ]; then
      last_progress=$SECONDS
    elif [ $((SECONDS - last_progress)) -ge "$stall_timeout" ]; then
      echo "ERROR: PGlite stalled — no progress for ${stall_timeout}s" >&2
      [ -f "$log_file" ] && echo "  last log:" >&2 && tail -3 "$log_file" >&2
      return 1
    fi

    sleep 1
  done

  echo "ERROR: PGlite not ready after ${max_timeout}s (hard ceiling)" >&2
  return 1
}

# ── Secrets (SOPS-encrypted .env.cai.enc) ─────────────────────
# Stub for M1: Aegir has no external credentials yet. Wire through so
# the structure is right when API keys / IAM tokens arrive in M2+.
if [ -x bin/bootstrap-secrets.sh ]; then
  bin/bootstrap-secrets.sh
fi
if [ -f .env.cai ]; then
  set -a
  . .env.cai
  set +a
fi

# ── Pre-flight: verify Python env is importable ───────────────
echo "Python: $(which python)"
echo "Packages: $(python -c 'import aegir; print("aegir OK")' 2>&1 || echo 'NOT FOUND')"

# ── Start PGlite if no external DB configured ─────────────────
if [ -z "$AEGIR_DB_URL" ] && [ -f scripts/pglite-server.mjs ]; then
  PGLITE_PORT=5545
  echo "Starting PGlite on port $PGLITE_PORT..."
  mkdir -p .app/pgdata
  PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=$PGLITE_PORT \
    node scripts/pglite-server.mjs > .app/pglite.log 2>&1 &
  PGLITE_PID=$!

  wait_for_pglite "$PGLITE_PORT" 10 120

  if ! kill -0 "$PGLITE_PID" 2>/dev/null; then
    echo "ERROR: PGlite process died during startup" >&2
    cat .app/pglite.log >&2
    exit 1
  fi

  # gssencmode=disable: psycopg sends GSSAPI negotiation before auth;
  # PGlite doesn't speak that. Omitting it yields:
  # "received invalid response to GSSAPI negotiation: R"
  export AEGIR_DB_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:${PGLITE_PORT}/postgres?sslmode=disable&gssencmode=disable"
fi

# ── Start Qdrant if vendored binary present ────────────────────
if [ -x qdrant/qdrant ]; then
  echo "Starting Qdrant on ports 6355/6356..."
  mkdir -p .app/qdrant/storage
  QDRANT__STORAGE__STORAGE_PATH=.app/qdrant/storage \
  QDRANT__SERVICE__HTTP_PORT=6355 \
  QDRANT__SERVICE__GRPC_PORT=6356 \
  qdrant/qdrant &
  wait_for_service "Qdrant" "curl -sf http://localhost:6355/healthz" 30
fi

# ── Resolve config (infra URLs now in environment) ────────────
python bin/resolve-config.py
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  # Strip surrounding double-quotes if our shell_quote wrote them.
  value="${value%\"}"; value="${value#\"}"
  export "$key=$value"
done < build/config/aegir.env

# ── Apply DB migrations ───────────────────────────────────────
# Idempotent; safe to run on every start.
echo "Running database bootstrap..."
echo "  DB URL: ${AEGIR_DB_URL:-<not set>}"
python -m aegir.db.bootstrap

# ── Start gateway (foreground) ────────────────────────────────
echo "Starting Aegir gateway on $HOST:$PORT..."
exec python -m aegir.gateway
