#!/usr/bin/env python3
"""Resolve Aegir's HOCON config from the current environment.

Reads ``config/base.conf`` with env-var substitution and writes hydrated build
artifacts for downstream consumers. Called by ``bin/start-app.sh`` (CAI),
``just resolve-config`` (local), and Zarf ``onDeploy`` actions (K8s).

Trusts whatever environment it's invoked in:
    devenv shell  → devenv.nix sets AEGIR_GATEWAY_PORT etc.
    CAI           → platform sets CDSW_APP_PORT; start-app.sh may set AEGIR_DB_URL
    Zarf K8s      → env from Kubernetes Secret + ConfigMap

Outputs:
    build/config/aegir.env   — flat KEY=value, shell-sourceable
    build/config/aegir.json  — structured JSON (for BDD assertions + conftest)
"""

from aegir.config import load_config, materialize_config, materialize_config_json

cfg = load_config()
materialize_config(cfg, "build/config/aegir.env")
materialize_config_json(cfg, "build/config/aegir.json")

print(
    f"[resolve-config] gateway={cfg.gateway.host}:{cfg.gateway.port} "
    f"db={cfg.db.url.split('@')[-1] if '@' in cfg.db.url else cfg.db.url} "
    f"qdrant={cfg.qdrant.host}:{cfg.qdrant.http_port} "
    f"runs={cfg.runs.dir}"
)
