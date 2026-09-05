#!/usr/bin/env bash
# Create the metaflow metadata DB in the aegir devenv postgres + the datastore bucket on
# Signals' RustFS. One-shot, idempotent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
check_disabled_exit DISABLE_METAFLOW "Metaflow"
banner "AEGIR METAFLOW DB SETUP"
wait_for_postgres "$USER"

echo "Ensuring metaflow user + database..."
psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d aegir -tc \
  "SELECT 1 FROM pg_roles WHERE rolname='metaflow'" | grep -q 1 || \
  psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d aegir -c "CREATE USER metaflow WITH PASSWORD 'metaflow'"
psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d aegir -tc \
  "SELECT 1 FROM pg_database WHERE datname='metaflow'" | grep -q 1 || \
  psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d aegir -c "CREATE DATABASE metaflow OWNER metaflow"
psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d metaflow -c \
  "GRANT ALL PRIVILEGES ON DATABASE metaflow TO metaflow" 2>/dev/null || true
echo "  metaflow user + database ready"

echo "Ensuring RustFS bucket (aegir-metaflow) on ${AEGIR_RUSTFS_ENDPOINT}..."
require_rustfs
mc alias set aegir "http://${AEGIR_RUSTFS_ENDPOINT}" "$AEGIR_RUSTFS_ACCESS_KEY" "$AEGIR_RUSTFS_SECRET_KEY" >/dev/null
mc mb --ignore-existing aegir/aegir-metaflow
echo "  RustFS bucket ready"
echo "Aegir Metaflow DB setup complete."
