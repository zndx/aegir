#!/usr/bin/env bash
# Create the metaflow metadata DB in the aegir devenv postgres + the MinIO bucket. One-shot, idempotent.
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

echo "Ensuring MinIO bucket (aegir-metaflow) on :${AEGIR_MINIO_PORT}..."
mc alias set aegir "http://localhost:${AEGIR_MINIO_PORT}" minioadmin minioadmin 2>/dev/null || true
mc mb --ignore-existing aegir/aegir-metaflow 2>/dev/null || true
echo "  MinIO bucket ready"
echo "Aegir Metaflow DB setup complete."
