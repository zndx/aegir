"""Aegir capability/gRPC engine server. Implements `Complete` (forwards to vLLM *inside* the engine via
the manager), plus admin `EnsureEndpoint`/`EngineStatus`. The vLLM endpoint is never exposed to callers.

    just engine-serve            # or: uv run --no-sync python -m aegir.engine.server
"""
from __future__ import annotations

import signal
import sys
from concurrent import futures

import grpc

from aegir.engine.config import ENGINE_GRPC_PORT
from aegir.engine.proto import aegir_engine_pb2 as pb
from aegir.engine.proto import aegir_engine_pb2_grpc as pbg
from aegir.engine.vllm_manager import VllmManager


class AegirEngineServicer(pbg.AegirEngineServicer):
    def __init__(self) -> None:
        self.mgr = VllmManager()

    def Complete(self, request, context):
        cap = request.capability or "instruct"
        try:
            out = self.mgr.complete(cap, request.prompt, request.system_prompt or "",
                                    request.max_tokens or 512, request.temperature or 0.7,
                                    json_schema=getattr(request, "json_schema", "") or "")
            return pb.CompleteResponse(
                text=out["text"], model=out["model"], prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"],
                reasoning_content=out["reasoning_content"], finish_reason=out["finish_reason"])
        except Exception as e:  # noqa: BLE001 — surface as a gRPC error, keep the engine up
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")

    def EnsureEndpoint(self, request, context):
        ep = self.mgr.ensure(request.capability or "instruct")
        return pb.EndpointStatus(capability=ep.capability, model=ep.spec.model, healthy=ep.healthy,
                                 port=ep.port, gpu_ids=ep.gpu_ids, detail=str(ep.log_path))

    def EngineStatus(self, request, context):
        eps = [pb.EndpointStatus(capability=e.capability, model=e.spec.model, healthy=e.healthy,
                                 port=e.port, gpu_ids=e.gpu_ids) for e in self.mgr.status()]
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return pb.EngineStatusResponse(endpoints=eps, total_gpus=n)


class ZndxEngineServicer:
    """The FEDERATION face — zndx.engine.v1.Engine (signals-protocol submodule), registered beside
    the native service so any signals engine's shared stub reaches us (the service-identity fix:
    gRPC method paths embed package+service, so wire-identical messages under per-project packages
    still get UNIMPLEMENTED — measured by Atelier on the first live cross-engine call, 2026-07-03).
    Delegates to the same VllmManager; engine-private details (internal vLLM ports) do not cross."""

    def __init__(self, mgr) -> None:
        self.mgr = mgr

    def Complete(self, request, context):
        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
        cap = request.capability or "instruct"
        try:
            out = self.mgr.complete(cap, request.prompt, request.system_prompt or "",
                                    request.max_tokens or 512, request.temperature or 0.7,
                                    json_schema=request.json_schema or "")
            return zpb.CompleteResponse(
                text=out["text"], model=out["model"], prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"],
                reasoning_content=out["reasoning_content"], finish_reason=out["finish_reason"])
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")

    def Status(self, request, context):
        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
        eps = [zpb.Endpoint(capability=e.capability, model=e.spec.model, healthy=e.healthy,
                            gpu_ids=e.gpu_ids) for e in self.mgr.status()]
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return zpb.StatusResponse(project="aegir", endpoints=eps, total_gpus=n)

    def Remediate(self, request, context):
        """Adapt to a boundary signal: compose the canonical signal+context into a re-authoring prompt,
        serve the adaptive inference (same vLLM path as Complete), and return the agent's proposed
        correction. The engine does NOT dispose it — the caller's membrane verifies + re-prompts. The
        prompt is composed HERE (server-side) so any federated caller sends only the structured signal."""
        import json
        import re
        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
        sig, ctx = request.signal, request.context
        kind = zpb.SignalKind.Name(sig.kind)
        cands = "\n".join(f"  - {c.iri}  \"{c.label}\" [{c.kind}]" for c in ctx.candidates) \
            or "  (none matched — no real external entity fits; you likely must coin an sdg: term)"
        anchors = "\n".join(f"  - {a.iri}  \"{a.label}\"" for a in ctx.anchors)
        justif = "\n".join(f"  - {j}" for j in ctx.justification)
        rules = "\n".join(f"  - {r}" for r in ctx.rules) \
            or "  - external namespaces (cco:/bfo:/fhir:) may NOT be coined; use a real IRI or coin sdg:."
        sysp = (
            "You are an ontology-authoring agent ADAPTING to a boundary signal: a reasoner/membrane detected "
            "an error in an axiom you must re-author. Reason about the CAUSE, then produce a single corrected "
            "Manchester axiom. Use ONLY real IRIs present in CANDIDATES or ANCHORS (they are from the CURRENT "
            "authoritative ontology); if none fits, COIN a term in the sdg: namespace. NEVER invent a cco:/bfo:/"
            "fhir: name. Output exactly one ```json block "
            "{\"correction\":\"<one Manchester axiom>\",\"disposition\":\"CORRECTED|COINED_LOCAL|UNRESOLVABLE\","
            "\"rationale\":\"<why>\"}.")
        prompt = (
            f"BOUNDARY SIGNAL [{kind}] — authority: {sig.authority}\n"
            f"  offending: {sig.offending}\n  reason: {sig.reason}\n\n"
            f"AXIOM TO RE-AUTHOR:\n  {sig.subject}\n\n"
            f"CANDIDATES (real entities from the current authority — reason about which, if any, is meant):\n{cands}\n\n"
            + (f"REASONER JUSTIFICATION:\n{justif}\n\n" if justif else "")
            + (f"DOMAIN ANCHORS:\n{anchors}\n\n" if anchors else "")
            + f"RULES:\n{rules}")
        schema = json.dumps({"type": "object", "properties": {
            "correction": {"type": "string"},
            "disposition": {"type": "string", "enum": ["CORRECTED", "COINED_LOCAL", "UNRESOLVABLE"]},
            "rationale": {"type": "string"}}, "required": ["correction", "disposition", "rationale"]})
        try:
            out = self.mgr.complete(request.capability or "instruct", prompt, sysp,
                                    request.max_tokens or 6000, request.temperature or 0.3, json_schema=schema)
            m = re.search(r"\{.*\}", out["text"], re.S)
            d = json.loads(m.group(0)) if m else {}
            dname = d.get("disposition", "UNRESOLVABLE")
            disp = zpb.Disposition.Value(dname) if dname in zpb.Disposition.keys() else zpb.UNRESOLVABLE
            return zpb.RemediationResponse(
                correction=d.get("correction", ""), disposition=disp, rationale=d.get("rationale", ""),
                model=out["model"], reasoning_content=out["reasoning_content"],
                completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"])
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"remediate[{kind}] failed: {e}")


def serve(port: int = ENGINE_GRPC_PORT) -> None:
    servicer = AegirEngineServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pbg.add_AegirEngineServicer_to_server(servicer, server)
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg
    zpbg.add_EngineServicer_to_server(ZndxEngineServicer(servicer.mgr), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"aegir-engine gRPC listening on :{port} — services: aegir.engine.AegirEngine + "
          f"zndx.engine.v1.Engine (federation face)", flush=True)

    def _stop(*_):
        servicer.mgr.shutdown()
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
