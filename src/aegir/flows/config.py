"""Metaflow config helpers — load ``config/metaflow/<mode>.json`` and apply to the environment
(lifted from Gaius). Call ``apply_metaflow_config(mode)`` BEFORE instantiating a FlowSpec.

Modes: ``rke2`` (service plane on the local RKE2 cluster — metadata service + the RustFS S3 datastore
(Signals', :9010) via port-forwards; the flow runs LOCALLY so steps keep GPU/engine/JVM/qdrant/postgres
access) and ``local``
(fully-local fallback: local metadata + local /raid datastore).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # aegir/src/aegir/flows/config.py → aegir/


def get_config_path(mode: str = "rke2") -> Path:
    return _PROJECT_ROOT / "config" / "metaflow" / f"{mode}.json"


def load_metaflow_config(mode: str = "rke2") -> dict:
    p = get_config_path(mode)
    return json.loads(p.read_text()) if p.exists() else {}


def apply_metaflow_config(mode: str = "rke2") -> None:
    """Set Metaflow env vars from the mode config (existing env vars win, so CLI overrides are honoured)."""
    for key, value in load_metaflow_config(mode).items():
        os.environ.setdefault(key, str(value))
