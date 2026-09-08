"""Ægir capability/gRPC engine — the federation face of a project that HOSTS NO MODEL.

**Strict layering:** workloads connect ONLY to this gRPC engine (`client.complete`) and name a
CAPABILITY, never a model or an endpoint. The engine FORWARDS inference capabilities (`instruct`,
`thinking`) over `zndx.engine.v1` to the peer that hosts the model (Gaius, Qwen3.8-27B) — never a
vLLM port, never an OpenAI-compatible URL. `instruct` is an operating profile of that model
(thinking on, effort low; signals-protocol capabilities.md). No local vLLM, no fallback
(2026-09-08; Qwen3.8-27B retired). Ægir still serves `Remediate`, `Announce` (peer
directory), `ServerQuery`, `Status` (endpoints empty — honest), and the KServe OIP face.
"""
