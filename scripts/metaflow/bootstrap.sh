#!/usr/bin/env bash
# One-shot K8s deploy of the aegir Metaflow service plane via `tilt ci` (idempotent).
# Renders the host-Endpoints + helm values with the RUNTIME devenv pg port ($PGPORT — non-deterministic),
# then deploys metaflow-service + UI into the aegir-metaflow namespace, pointed at the devenv postgres+MinIO.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib.sh"
check_disabled_exit DISABLE_METAFLOW "Metaflow"
banner "AEGIR METAFLOW BOOTSTRAP — K8s deploy (ns: $AEGIR_MF_NAMESPACE)"

if ! kubectl cluster-info &>/dev/null; then
  echo "ERROR: cannot reach RKE2 (KUBECONFIG=$KUBECONFIG). Is rke2-server up?"; exit 1
fi
kubectl get namespace "$AEGIR_MF_NAMESPACE" &>/dev/null || kubectl create namespace "$AEGIR_MF_NAMESPACE"

if kubectl get pods -n "$AEGIR_MF_NAMESPACE" -l app.kubernetes.io/name=metaflow-service \
     --no-headers 2>/dev/null | grep -q Running; then
  echo "  metaflow-service already running in $AEGIR_MF_NAMESPACE — skipping deploy"; exit 0
fi

# Render templates with the runtime pg port + host IP into build/metaflow/ (gitignored).
RENDER_DIR="$REPO/build/metaflow"
mkdir -p "$RENDER_DIR"
export PGPORT AEGIR_HOST_IP AEGIR_MINIO_PORT AEGIR_MF_NAMESPACE
for t in aegir-services.yaml metaflow-service-values.yaml metaflow-ui-values.yaml; do
  src="$REPO/infra/tilt/$t"; [ -f "$src" ] || src="$REPO/infra/k8s/$t"
  envsubst < "$src" > "$RENDER_DIR/$t"
done
echo "  rendered Endpoints+values for pg :$PGPORT, minio :$AEGIR_MINIO_PORT, host $AEGIR_HOST_IP"

echo "Deploying via tilt ci..."
cd "$REPO/infra/tilt"
AEGIR_RENDER_DIR="$RENDER_DIR" tilt ci --timeout 420s -- --namespace="$AEGIR_MF_NAMESPACE"
echo "  metaflow service plane deployed to $AEGIR_MF_NAMESPACE"
