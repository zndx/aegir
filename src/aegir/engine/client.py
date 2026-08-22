"""Thin capability-client — the ONLY way a workload reaches the engine: `complete(prompt, capability=…)`
→ text, over gRPC. No vLLM endpoint is ever returned to the caller (strict layering). Federation:
setting AEGIR_ENGINE_FEDERATE delegates capability requests to a remote (e.g. Gaius) engine; the
client negotiates the peer's protocol face down a ladder — native → zndx.engine.v1 → KServe OIP —
falling through on UNIMPLEMENTED (gRPC's unknown-service answer: method paths embed package+service,
so a wire-identical message under the wrong service name still bounces — the measured Atelier
finding) and caching what answered per target. Every face returns the SAME dict shape, so callers
never know which protocol served them. Pin with AEGIR_ENGINE_FEDERATE_PROTOCOL to skip the probe.
"""
from __future__ import annotations

import grpc

from aegir.engine.config import ENGINE_GRPC_PORT, FEDERATE_PROTOCOL, FEDERATE_TARGET
from aegir.engine.proto import aegir_engine_pb2 as pb
from aegir.engine.proto import aegir_engine_pb2_grpc as pbg

_negotiated: dict[str, str] = {}  # target -> the protocol face that answered


def _target() -> str:
    return FEDERATE_TARGET or f"127.0.0.1:{ENGINE_GRPC_PORT}"


def _complete_native(ch, cap, prompt, system_prompt, max_tokens, temperature, timeout, json_schema):
    req = pb.CompleteRequest(capability=cap, prompt=prompt, system_prompt=system_prompt,
                             max_tokens=max_tokens, temperature=temperature)
    if json_schema:
        req.json_schema = json_schema
    r = pbg.AegirEngineStub(ch).Complete(req, timeout=timeout)
    return {"text": r.text, "reasoning_content": r.reasoning_content, "finish_reason": r.finish_reason,
            "model": r.model, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms}


def _complete_zndx(ch, cap, prompt, system_prompt, max_tokens, temperature, timeout, json_schema):
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as z
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zg
    r = zg.EngineStub(ch).Complete(
        z.CompleteRequest(capability=cap, prompt=prompt, system_prompt=system_prompt,
                          max_tokens=max_tokens, temperature=temperature, json_schema=json_schema),
        timeout=timeout)
    return {"text": r.text, "reasoning_content": r.reasoning_content, "finish_reason": r.finish_reason,
            "model": r.model, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms}


def _complete_oip(ch, cap, prompt, system_prompt, max_tokens, temperature, timeout, json_schema):
    """Lower a capability request to KServe OIP ModelInfer (the inverse of the server's OIP face,
    same Gaius wire conventions — capability = model name, BYTES text tensors, params for the rest)."""
    from aegir.engine.proto import open_inference_grpc_pb2 as opb
    from aegir.engine.proto import open_inference_grpc_pb2_grpc as opbg
    req = opb.ModelInferRequest(model_name=cap)
    for name, val in (("prompt", prompt), ("system_prompt", system_prompt)):
        if not val:
            continue
        t = req.inputs.add()
        t.name, t.datatype = name, "BYTES"
        t.shape.extend([1])
        t.contents.bytes_contents.append(val.encode("utf-8"))
    req.parameters["max_tokens"].int64_param = max_tokens
    req.parameters["temperature"].double_param = temperature
    if json_schema:
        req.parameters["json_schema"].string_param = json_schema
    r = opbg.GRPCInferenceServiceStub(ch).ModelInfer(req, timeout=timeout)
    outs = {o.name: (o.contents.bytes_contents[0].decode("utf-8") if o.contents.bytes_contents else "")
            for o in r.outputs}
    p = r.parameters
    return {"text": outs.get("completion", ""), "reasoning_content": outs.get("reasoning_content", ""),
            "finish_reason": p["finish_reason"].string_param if "finish_reason" in p else "",
            "model": p["model"].string_param if "model" in p else "",
            "prompt_tokens": int(p["prompt_tokens"].int64_param) if "prompt_tokens" in p else 0,
            "completion_tokens": int(p["tokens_used"].int64_param) if "tokens_used" in p else 0,
            "latency_ms": float(p["latency_ms"].double_param) if "latency_ms" in p else 0.0}


_LADDER = (("native", _complete_native), ("zndx", _complete_zndx), ("oip", _complete_oip))


def complete_detailed(prompt: str, *, capability: str = "instruct", system_prompt: str = "",
                      max_tokens: int = 4096, temperature: float = 0.7, timeout: float = 1200.0,
                      json_schema: str = "") -> dict:
    """Like `complete`, but returns the full response — including the retained `reasoning_content`
    (the thinking trace, a corpus value-add) and `finish_reason` ('length' ⇒ raise max_tokens). The
    generous default max_tokens lets verbose reasoning traces finish; the workload accepts the wait.
    ``json_schema`` (a JSON Schema string) requests engine-enforced structured output (vLLM
    guided-json) — the cross-engine convergence field; empty = unconstrained."""
    target = _target()
    if FEDERATE_PROTOCOL:  # explicit pin: exactly this face, no probe, no fallback
        by_name = dict(_LADDER)
        if FEDERATE_PROTOCOL not in by_name:
            raise ValueError(f"unknown AEGIR_ENGINE_FEDERATE_PROTOCOL {FEDERATE_PROTOCOL!r}; "
                             f"known: {list(by_name)}")
        ladder = [(FEDERATE_PROTOCOL, by_name[FEDERATE_PROTOCOL])]
    else:  # negotiated face first (stale after a peer redeploy ⇒ falls through and re-caches)
        cached = _negotiated.get(target)
        ladder = sorted(_LADDER, key=lambda nf: nf[0] != cached)
    with grpc.insecure_channel(target) as ch:
        for i, (name, fn) in enumerate(ladder):
            try:
                out = fn(ch, capability, prompt, system_prompt, max_tokens, temperature,
                         timeout, json_schema)
            except grpc.RpcError as e:
                # UNIMPLEMENTED = this face isn't registered on the peer — try the next rung.
                # Anything else (NOT_FOUND, INTERNAL, DEADLINE_EXCEEDED) is a real answer: raise.
                if e.code() == grpc.StatusCode.UNIMPLEMENTED and i + 1 < len(ladder):
                    continue
                raise
            _negotiated[target] = name
            return out
    raise AssertionError("unreachable")  # the last rung either returned or raised


def complete(prompt: str, *, capability: str = "instruct", system_prompt: str = "",
             max_tokens: int = 4096, temperature: float = 0.7, timeout: float = 1200.0) -> str:
    """Request the `capability` to complete `prompt`; returns the answer text. Thinking is RETAINED —
    for a model that embeds its trace inline (no parseable delimiter) the reasoning is part of this text;
    use `complete_detailed` to get the separated `reasoning_content` when available. (Long default
    timeout: first call may trigger an on-demand vLLM load + a verbose trace.)"""
    return complete_detailed(prompt, capability=capability, system_prompt=system_prompt,
                             max_tokens=max_tokens, temperature=temperature, timeout=timeout)["text"]


def remediate(signal: dict, context: dict, *, capability: str = "reauthor", max_tokens: int = 12000,
              temperature: float = 0.3, timeout: float = 1200.0) -> dict:
    """Adapt to a boundary SIGNAL via the engine's Remediate capability (Holland CAS). The engine reasons
    over the signal + LIVE context and returns a proposed correction; the CALLER's membrane disposes it +
    re-prompts. Uses the FEDERATION face (zndx.engine.v1) — the same shape a remote Atelier boundary would.

    ``signal`` = {kind ('EXTERNAL_NAMESPACE_VIOLATION'|'UNSATISFIABLE'|'UNGROUNDED'|'VERSION_DRIFT'),
    subject (the axiom), offending, reason, authority}. ``context`` = {candidates:[{iri,label,kind,score}],
    justification:[str], anchors:[{iri,label,...}], rules:[str]} — candidates queried LIVE from the current
    authority, never a baked mapping. Returns {correction, disposition, rationale, model, reasoning_content, …}."""
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as z
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zg

    def _cands(lst):
        return [z.Candidate(iri=c.get("iri", ""), label=c.get("label", ""), kind=c.get("kind", ""),
                            score=float(c.get("score", 0.0))) for c in (lst or [])]

    sig = z.BoundarySignal(kind=z.SignalKind.Value(signal.get("kind", "SIGNAL_KIND_UNSPECIFIED")),
                           subject=signal.get("subject", ""), offending=signal.get("offending", ""),
                           reason=signal.get("reason", ""), authority=signal.get("authority", ""))
    ctx = z.SignalContext(candidates=_cands(context.get("candidates")),
                          justification=list(context.get("justification", [])),
                          anchors=_cands(context.get("anchors")), rules=list(context.get("rules", [])))
    with grpc.insecure_channel(_target()) as ch:
        r = zg.EngineStub(ch).Remediate(
            z.RemediationRequest(capability=capability, signal=sig, context=ctx,
                                 max_tokens=max_tokens, temperature=temperature), timeout=timeout)
        return {"correction": r.correction, "disposition": z.Disposition.Name(r.disposition),
                "rationale": r.rationale, "model": r.model, "reasoning_content": r.reasoning_content,
                "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms}


def engine_status(timeout: float = 10.0):
    with grpc.insecure_channel(_target()) as ch:
        return pbg.AegirEngineStub(ch).EngineStatus(pb.EngineStatusRequest(), timeout=timeout)
