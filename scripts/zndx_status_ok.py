#!/usr/bin/env python3
"""Return 0 if zndx.engine.v1.Engine/Status reports project=aegir.

Uses generated signals-protocol stubs under src/aegir/engine/proto
(the proto is the specification — implement via codegen).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src",):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import grpc  # noqa: E402
from aegir.engine.proto.zndx.engine.v1 import engine_pb2, engine_pb2_grpc  # noqa: E402


def main() -> int:
    port = os.environ.get("AEGIR_ENGINE_PORT", "50151")
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = engine_pb2_grpc.EngineStub(channel)
    try:
        r = stub.Status(engine_pb2.StatusRequest(), timeout=3)
    except Exception as e:  # noqa: BLE001
        print(f"Status failed: {e}", file=sys.stderr)
        return 1
    if r.project != "aegir":
        print(f"project={r.project!r} expected aegir", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
