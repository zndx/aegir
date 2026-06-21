#!/usr/bin/env bash
# Interactive Tilt dashboard for the aegir Metaflow stack (opt-in: `devenv processes up metaflow-ui`).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib.sh"
banner "AEGIR METAFLOW UI — Tilt dashboard"
echo "  Tilt dashboard: http://localhost:10351   Metaflow UI: http://localhost:3001"
cd "$REPO/infra/tilt"
AEGIR_RENDER_DIR="$REPO/build/metaflow" exec tilt up --port 10351 --stream -- --namespace="$AEGIR_MF_NAMESPACE"
