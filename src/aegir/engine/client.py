"""Thin capability-client — the ONLY way a workload reaches the engine: `complete(prompt, capability=…)`
→ text, over gRPC. No vLLM endpoint is ever returned to the caller (strict layering). Federation:
setting AEGIR_ENGINE_FEDERATE delegates capability requests to a remote (e.g. Gaius) engine — roadmap.
"""
from __future__ import annotations

import grpc

from aegir.engine.config import ENGINE_GRPC_PORT, FEDERATE_TARGET
from aegir.engine.proto import aegir_engine_pb2 as pb
from aegir.engine.proto import aegir_engine_pb2_grpc as pbg


def _target() -> str:
    return FEDERATE_TARGET or f"127.0.0.1:{ENGINE_GRPC_PORT}"


def complete_detailed(prompt: str, *, capability: str = "instruct", system_prompt: str = "",
                      max_tokens: int = 4096, temperature: float = 0.7, timeout: float = 1200.0,
                      json_schema: str = "") -> dict:
    """Like `complete`, but returns the full response — including the retained `reasoning_content`
    (the thinking trace, a corpus value-add) and `finish_reason` ('length' ⇒ raise max_tokens). The
    generous default max_tokens lets verbose reasoning traces finish; the workload accepts the wait.
    ``json_schema`` (a JSON Schema string) requests engine-enforced structured output (vLLM
    guided-json) — the cross-engine convergence field; empty = unconstrained."""
    with grpc.insecure_channel(_target()) as ch:
        req = pb.CompleteRequest(
            capability=capability, prompt=prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, temperature=temperature)
        if json_schema:
            req.json_schema = json_schema
        r = pbg.AegirEngineStub(ch).Complete(req, timeout=timeout)
        return {"text": r.text, "reasoning_content": r.reasoning_content, "finish_reason": r.finish_reason,
                "model": r.model, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "latency_ms": r.latency_ms}


def complete(prompt: str, *, capability: str = "instruct", system_prompt: str = "",
             max_tokens: int = 4096, temperature: float = 0.7, timeout: float = 1200.0) -> str:
    """Request the `capability` to complete `prompt`; returns the answer text. Thinking is RETAINED —
    for a model that embeds its trace inline (no parseable delimiter) the reasoning is part of this text;
    use `complete_detailed` to get the separated `reasoning_content` when available. (Long default
    timeout: first call may trigger an on-demand vLLM load + a verbose trace.)"""
    return complete_detailed(prompt, capability=capability, system_prompt=system_prompt,
                             max_tokens=max_tokens, temperature=temperature, timeout=timeout)["text"]


def engine_status(timeout: float = 10.0):
    with grpc.insecure_channel(_target()) as ch:
        return pbg.AegirEngineStub(ch).EngineStatus(pb.EngineStatusRequest(), timeout=timeout)
