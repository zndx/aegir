#!/usr/bin/env bash
# Expose the aegir Metaflow service plane: patch metaflow-service to NodePort 30181/30183 + forward the UI.
# Coexistence: aegir uses 30181/30183 + UI 3001/9085 (gaius uses 30180/30182 + 3000/9083).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
check_disabled DISABLE_METAFLOW "Metaflow"
banner "AEGIR METAFLOW PORT FORWARDS"
NS="$AEGIR_MF_NAMESPACE"

echo "Waiting for metaflow-service pod in $NS..."
for i in $(seq 1 60); do
  kubectl get pods -n "$NS" -l app.kubernetes.io/name=metaflow-service --no-headers 2>/dev/null \
    | grep -q Running && break
  [ "$i" -eq 60 ] && { echo "ERROR: metaflow-service not running after 300s"; exit 1; }
  echo "  waiting... (${i}x5s)"; sleep 5
done
kubectl wait -n "$NS" --for=condition=ready pod \
  -l app.kubernetes.io/name=metaflow-ui-static --timeout=120s 2>/dev/null || true
echo "  pods ready"

echo "Patching metaflow-service → NodePort 30181/30183..."
kubectl patch svc metaflow-service -n "$NS" -p \
  '{"spec":{"type":"NodePort","ports":[{"name":"metadata","port":8080,"targetPort":8080,"nodePort":30181},{"name":"upgrades","port":8082,"targetPort":8082,"nodePort":30183}]}}' \
  2>/dev/null || true
echo "  Metaflow Service: http://localhost:30181"

pkill -f "kubectl.*port-forward.*3001:3000" 2>/dev/null || true
pkill -f "kubectl.*port-forward.*9085:8083" 2>/dev/null || true
sleep 1
echo "UI: http://0.0.0.0:3001  ·  UI API: http://0.0.0.0:9085"
while true; do
  kubectl port-forward -n "$NS" --address 0.0.0.0 svc/metaflow-ui-static 3001:3000 &
  PF1=$!
  kubectl port-forward -n "$NS" --address 0.0.0.0 svc/metaflow-ui 9085:8083 &
  PF2=$!
  wait -n $PF1 $PF2 2>/dev/null || true
  echo "UI port-forward exited, restarting in 5s..."; kill $PF1 $PF2 2>/dev/null || true; sleep 5
done
