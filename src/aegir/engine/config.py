"""Engine config — the capability vocabulary Ægir speaks and how its engine reaches the federation.

Ægir HOSTS NO MODEL (2026-09-08). Inference capabilities (``instruct``, ``thinking``, …) are served by
the federation peer that hosts the model — today Gaius, Qwen3.8-27B — and reached over
``zndx.engine.v1``; this engine FORWARDS (``forwarder.py``). ``instruct`` is an OPERATING PROFILE of
that model (thinking on, ``reasoning_effort=low``; signals-protocol ``capabilities.md`` §Operating
profiles) — the SERVING engine aligns its call parameters; a caller only names the capability.
No local vLLM, no fallback: a capability nobody advertises fails fast (FAILED_PRECONDITION).

Strict layering is unchanged: workloads speak only to THIS engine (``client.complete``); this engine
speaks only the protocol. Qwen3.6-35B-A3B-FP8 and the local vLLM manager are retired.
"""
from __future__ import annotations

import os

ENGINE_GRPC_PORT = int(os.environ.get("AEGIR_ENGINE_PORT", "50151"))

# Client-side delegation (workload → a REMOTE engine instead of the local one). The client negotiates
# the peer's protocol face (native → zndx.engine.v1 → KServe OIP); AEGIR_ENGINE_FEDERATE_PROTOCOL pins
# one ("native" | "zndx" | "oip") and skips the probe.
FEDERATE_TARGET = os.environ.get("AEGIR_ENGINE_FEDERATE", "")
FEDERATE_PROTOCOL = os.environ.get("AEGIR_ENGINE_FEDERATE_PROTOCOL", "")

# The capability a request names when it leaves `capability` empty. Ægir's inference class is
# `instruct`: the same resident model as `thinking`, a brief trace, less overhead.
DEFAULT_CAPABILITY = "instruct"

# Remediate composes an ontology re-authoring prompt server-side and needs the FULL trace to reason
# about a reasoner's justification — that is `thinking`, not the brief `instruct` profile.
REMEDIATE_CAPABILITY = "thinking"

# Capabilities the native EngineStatus resolves proactively so `just engine-ready` / readiness can
# report a route (project-internal face; the lattice Status lists only what is HOSTED — nothing).
ROUTE_CAPABILITIES: tuple[str, ...] = ("instruct", "thinking")

PEER_STATUS_TTL_S = float(os.environ.get("AEGIR_PEER_STATUS_TTL", "30"))
PEER_STATUS_TIMEOUT_S = float(os.environ.get("AEGIR_PEER_STATUS_TIMEOUT", "5"))
FORWARD_TIMEOUT_S = float(os.environ.get("AEGIR_FORWARD_TIMEOUT", "1200"))
