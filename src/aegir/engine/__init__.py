"""Aegir capability/gRPC engine — the local LLM substrate (mirrors the Gaius engine pattern).

**Strict layering:** the engine is the SOLE vLLM client and owns the capability→model mapping;
**workloads connect ONLY to the gRPC engine** (`client.complete`) — never to vLLM, never handed an
endpoint URL. Serves Qwen3.6-35B-A3B-FP8 across the local GPUs (tensor-parallel). Federation with the
Gaius engine is a roadmap config flag (`AEGIR_ENGINE_FEDERATE`).
"""
