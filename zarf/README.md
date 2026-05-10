# Aegir — Zarf air-gap deployment

This directory packages Aegir's leaderboard gateway + UI for air-gap
Kubernetes deployment via Zarf. It mirrors
`/home/rch/local/src/cldr/cybersec/zarf/` in structure, scoped down to
Aegir's M1 surface: FastAPI gateway + Postgres 16 + Qdrant + ingress.

## What's in the package

| Component | Purpose |
|-----------|---------|
| `local-path-provisioner` | Default StorageClass for bare RKE2 (skippable on EKS / Cloudera ECS) |
| `aegir-images` | Pulls: aegir-gateway, pgvector/pgvector:pg16, qdrant/qdrant |
| `aegir-namespace` | Creates the `aegir` namespace |
| `postgres` | pgvector-enabled Postgres 16 StatefulSet + PVC |
| `qdrant` | Qdrant StatefulSet (empty in M1; reserved for M2+) |
| `aegir-gateway` | FastAPI + static React bundle Deployment + PVC for runs sidecars |
| `ingress` | Traefik Ingress at `aegir.{domain}` |
| `cloudera-aiis-annotations` | Optional overlay for Cloudera AI Inference Service |

## Build

From the repo root::

    just zarf-build

equivalent to::

    # 1. Build the gateway image
    podman build -t localhost:5555/aegir-gateway:0.1.0 \
        -f zarf/images/Dockerfile.aegir-gateway .

    # 2. Start podman socket so Zarf can inspect images
    podman system service --time=600 unix:///run/user/$(id -u)/podman/podman.sock &

    # 3. Create the Zarf package
    cd zarf
    DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock \
        zarf package create . --confirm --skip-sbom

Output: `zarf-package-aegir-leaderboard-amd64-0.1.0.tar.zst` (~1 GB — mostly
Python image layers + Postgres/Qdrant images).

## Transfer to air-gap

The resulting `.tar.zst` is a single file, suitable for physical transfer
(USB, data diode) to the destination air-gap network. No runtime network
dependencies once transferred.

## Deploy

AWS EKS / generic K8s::

    zarf package deploy zarf-package-aegir-leaderboard-*.tar.zst --confirm \
        --set INGRESS_DOMAIN=aegir.example.com \
        --set POSTGRES_PASSWORD=... \
        --set GATEWAY_REPLICAS=2

Cloudera AI Inference Service::

    zarf package deploy zarf-package-aegir-leaderboard-*.tar.zst --confirm \
        --set TARGET=cloudera-aiis \
        --set INGRESS_DOMAIN=aegir.cai.acme.com \
        --set POSTGRES_PASSWORD=...

The `TARGET=cloudera-aiis` variable activates the
`cloudera-aiis-annotations` component, which overlays Cloudera-specific
workspace labels and Ranger-resource-tag annotations onto the gateway
Deployment. Stock-K8s deploys leave that component disabled.

## Verify

After deploy::

    kubectl get pods -n aegir
    kubectl get svc -n aegir
    kubectl get ingress -n aegir
    # Tail gateway logs:
    kubectl logs -n aegir -l app=aegir-gateway -f
    # API health from inside the cluster:
    kubectl exec -n aegir deploy/aegir-gateway -- curl -s localhost:8091/api/health

## What's not here yet (M2+)

- KServe `InferenceService` predictor for online Aegir inference (M3+)
- `aegir-ui` as a separate image (currently the gateway serves ui/dist
  itself — simpler for M1)
- GPU-flavored training image for producing runs in-cluster (M3+)
- External-baseline harness manifests (Nemotron / OpenAI OSS) (M2)
