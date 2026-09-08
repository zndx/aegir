"""Ægir capability/gRPC engine server — hosts NO model; forwards inference to the federation.

Three faces on one port: the native ``aegir.engine.AegirEngine`` (project-internal), the shared
``zndx.engine.v1.Engine`` (federation), and KServe OIP. ``Complete`` on every face forwards the
capability request over ``zndx.engine.v1`` to the peer whose Status serves it (Gaius today; see
``forwarder.py``) — never a vLLM port, never an OpenAI URL. The lattice ``Status`` advertises NO
endpoints (nothing hosted — honest); the native ``EngineStatus`` lists the resolved federation
ROUTES so readiness / ``just engine-ready`` still have a signal.

    just engine-serve            # or: uv run --no-sync python -m aegir.engine.server
"""
from __future__ import annotations

import signal
import sys
from concurrent import futures

import grpc

from aegir.engine.config import DEFAULT_CAPABILITY, ENGINE_GRPC_PORT, REMEDIATE_CAPABILITY
from aegir.engine.forwarder import CapabilityForwarder, NoPeerServes
from aegir.engine.proto import aegir_engine_pb2 as pb
from aegir.engine.proto import aegir_engine_pb2_grpc as pbg


class AegirEngineServicer(pbg.AegirEngineServicer):
    """The project-internal face. ``mgr`` is the CapabilityForwarder (the retired VllmManager's
    surface: ``complete`` / ``status`` / ``shutdown``) so the other faces share one resolver."""

    def __init__(self) -> None:
        self.mgr = CapabilityForwarder()

    def Complete(self, request, context):
        cap = request.capability or DEFAULT_CAPABILITY
        try:
            out = self.mgr.complete(cap, request.prompt, request.system_prompt or "",
                                    request.max_tokens or 512, request.temperature or 0.7,
                                    json_schema=getattr(request, "json_schema", "") or "")
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except grpc.RpcError as e:  # the serving peer's answer IS the answer — no retry, no fallback
            context.abort(e.code(), f"complete[{cap}] refused by the serving peer: {e.details()}")
        except Exception as e:  # noqa: BLE001 — surface as a gRPC error, keep the engine up
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")
        return pb.CompleteResponse(
            text=out["text"], model=out["model"], prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"],
            reasoning_content=out["reasoning_content"], finish_reason=out["finish_reason"])

    def EnsureEndpoint(self, request, context):
        cap = request.capability or DEFAULT_CAPABILITY
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"aegir hosts no models — nothing to ensure for {cap!r}; inference is forwarded to the "
            f"federation peer whose Status serves it (see EngineStatus for the resolved routes)")

    def EngineStatus(self, request, context):
        """Native face: the resolved federation ROUTES (not hosted endpoints — those are none)."""
        eps = [pb.EndpointStatus(capability=r.capability, model=r.model, healthy=True, port=0,
                                 gpu_ids=list(r.gpu_ids), detail=f"forwarded → {r.label}")
               for r in self.mgr.routes()]
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            n = 0
        return pb.EngineStatusResponse(endpoints=eps, total_gpus=n)


PROJECT = "aegir"
# Service names advertised via gRPC reflection (lattice-ci external grpcurl).
LATTICE_SERVICE_NAMES = (
    "aegir.engine.AegirEngine",
    "zndx.engine.v1.Engine",
    "inference.GRPCInferenceService",
)


def build_status_response(mgr):
    """Project engine state onto zndx.engine.v1.StatusResponse — the lattice face.

    Ægir HOSTS NO MODEL: ``endpoints`` is what this engine serves on its own GPUs — nothing. An
    engine that forwards a capability MUST NOT advertise it (capabilities.md §Operating profiles),
    so there is no placeholder row; Signals' peer contract carries no capability hint for aegir and
    lattice-ci accepts project=aegir at gRPC bind. ``mgr.status()`` is kept for symmetry and is
    empty by construction.
    """
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb

    eps = [zpb.Endpoint(capability=e.capability, model=e.spec.model, healthy=e.healthy,
                        gpu_ids=list(e.gpu_ids))
           for e in mgr.status()]
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 — Status must answer even without torch/CUDA
        n = 0
    from aegir.engine.s2s import local_surfaces
    return zpb.StatusResponse(
        project=PROJECT, endpoints=eps, total_gpus=n, surfaces=local_surfaces())


def enable_reflection(server) -> None:
    """Enable gRPC server reflection; required on the lattice port.

    Bare ``grpcurl -plaintext host:port list`` / ``Engine/Status`` must work
    without local descriptors (signals-protocol engine_grpc.md).
    """
    from grpc_reflection.v1alpha import reflection
    reflection.enable_server_reflection(
        (*LATTICE_SERVICE_NAMES, reflection.SERVICE_NAME), server)


class ZndxEngineServicer:
    """The FEDERATION face — zndx.engine.v1.Engine (signals-protocol submodule), registered beside
    the native service so any signals engine's shared stub reaches us (the service-identity fix:
    gRPC method paths embed package+service, so wire-identical messages under per-project packages
    still get UNIMPLEMENTED — measured by Atelier on the first live cross-engine call, 2026-07-03).
    Delegates to the same CapabilityForwarder; nothing engine-private crosses (there is no local vLLM)."""

    def __init__(self, mgr) -> None:
        self.mgr = mgr

    def Complete(self, request, context):
        """Forward VERBATIM (tools_json / messages_json / capabilities[] ride along) to the peer whose
        Status serves the capability. The peer aligns to the capability's operating profile and
        reports it (``profile``); this engine adds nothing and falls back to nothing."""
        cap = request.capability or DEFAULT_CAPABILITY
        try:
            return self.mgr.forward(request)
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except grpc.RpcError as e:
            context.abort(e.code(), f"complete[{cap}] refused by the serving peer: {e.details()}")
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")

    def Status(self, request, context):
        return build_status_response(self.mgr)

    def Remediate(self, request, context):
        """Adapt to a boundary signal: compose the canonical signal+context into a re-authoring prompt,
        serve the adaptive inference (forwarded like Complete), and return the agent's proposed
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
            "Manchester axiom. STRONGLY PREFER a real IRI from CANDIDATES/ANCHORS (they are the CURRENT "
            "authoritative BFO/CCO entities) — choose the one whose meaning fits, and for a RELATION "
            "(object-property) slot pick the correct DIRECTION: a WHOLE 'has continuant part' (bfo:0000178) its "
            "parts; a PART is 'continuant part of' (bfo:0000176) its whole. Almost every structural relation "
            "already exists in BFO/CCO — reach for it. COIN in the sdg: namespace ONLY when NO real entity can "
            "express the meaning; if you coin an sdg: OBJECT PROPERTY you MUST ground it in the SAME axiom set by "
            "also emitting `ObjectProperty: sdg:<name> SubPropertyOf: <a real bfo:/cco: relation>` (an ungrounded "
            "coined relation is rejected). NEVER invent a cco:/bfo:/fhir: name. Set disposition CORRECTED when you "
            "used a real IRI, COINED_LOCAL only when you had to coin (and grounded it). Output exactly one ```json "
            "block {\"correction\":\"<one or more Manchester frames>\",\"disposition\":\"CORRECTED|COINED_LOCAL|"
            "UNRESOLVABLE\",\"rationale\":\"<why>\"}.")
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
        # `request.capability` ("reauthor") names Ægir's REMEDIATION class; the INFERENCE it needs is the
        # full-trace `thinking` profile — a re-authoring decision reasons over a reasoner's justification.
        try:
            out = self.mgr.complete(REMEDIATE_CAPABILITY, prompt, sysp,
                                    request.max_tokens or 12000, request.temperature or 0.3, json_schema=schema)
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            return
        except grpc.RpcError as e:
            context.abort(e.code(), f"remediate[{kind}] refused by the serving peer: {e.details()}")
            return
        except Exception as e:  # noqa: BLE001 — a genuine engine failure surfaces as a gRPC error
            context.abort(grpc.StatusCode.INTERNAL, f"remediate[{kind}] failed: {e}")
            return
        # A truncated/malformed completion is an UNRESOLVABLE disposition, NOT an RPC failure: the caller's
        # membrane must still receive a well-formed response so one bad template cannot crash a batch sweep.
        m = re.search(r"\{.*\}", out["text"], re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except ValueError:
            d = {}
        if not d:
            trunc = " (output truncated — raise max_tokens)" if out.get("finish_reason") == "length" else ""
            return zpb.RemediationResponse(
                correction="", disposition=zpb.UNRESOLVABLE,
                rationale=f"model output was not valid JSON{trunc}: {out['text'][:400]}",
                model=out["model"], reasoning_content=out.get("reasoning_content", ""),
                completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"])
        dname = d.get("disposition", "UNRESOLVABLE")
        disp = zpb.Disposition.Value(dname) if dname in zpb.Disposition.keys() else zpb.UNRESOLVABLE
        return zpb.RemediationResponse(
            correction=d.get("correction", ""), disposition=disp, rationale=d.get("rationale", ""),
            model=out["model"], reasoning_content=out["reasoning_content"],
            completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"])

    def Yield(self, request, context):
        """C2 → engine. Aegir has no sentinel workloads; unknown id is idempotent."""
        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
        return zpb.YieldResponse(
            ok=True, process_ended=False, restore_started=False,
            message="aegir has no sentinel workloads")

    def ServerQuery(self, request, context):
        from aegir.engine.s2s import local_response
        return local_response(int(request.kind))

    def RecordLineage(self, request, context):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "RecordLineage is Signals Atlas SoR (POST /api/v1/lineage).")

    def WatchWorkload(self, request, context):
        """Held-open intended serving set. Ægir hosts no model: intents are intentionally empty."""
        import time as _time

        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb

        generation = 0
        while context.is_active():
            yield zpb.WorkloadProfile(
                phase=zpb.WORKLOAD_PHASE_SETTLED,
                generation=generation,
                intents=[],
                settled_at_unix_ms=int(_time.time() * 1000),
                detail="aegir hosts no model — intents intentionally empty; inference is forwarded to the federation",
            )
            _time.sleep(15)
            generation += 1

    def Announce(self, request, context):
        """TTL'd S2S join — remember a PeerHint in this engine's PEERS set."""
        from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
        from aegir.engine.s2s import remember_announce

        ok, ttl, err = remember_announce(
            project=request.project,
            engine_target=request.engine_target,
            ttl_seconds=int(request.ttl_seconds or 0),
        )
        return zpb.AnnounceAck(accepted=ok, ttl_seconds=ttl, error=err)


class OipInferenceServicer:
    """The STANDARD face — inference.GRPCInferenceService (KServe Open Inference Protocol),
    registered beside the native + zndx services so OIP-native peers (Gaius chose OIP as its peer
    protocol; vanilla Triton/TGI/CIS clients) reach us without speaking zndx.engine.v1. Lowering per
    engine_grpc.md §"OIP mapping": capability names ARE model names; Complete lowers to ModelInfer
    with a text tensor. Tensor conventions mirror Gaius's servicer exactly (BYTES "prompt" /
    "system_prompt" inputs, max_tokens/temperature request parameters, BYTES "completion" output,
    metadata as response parameters) — that symmetry is the interop contract, don't drift it.
    Additive extras a Gaius client simply ignores: a "json_schema" string_param (engine-enforced
    structured output, same as the other faces) and a "reasoning_content" output tensor when the
    model separates its trace (retained, never dropped — traces are corpus value-adds)."""

    def __init__(self, mgr) -> None:
        self.mgr = mgr

    @staticmethod
    def _bytes_input(request, name: str) -> str:
        """First element of a BYTES input tensor, from either OIP representation: `contents`
        (what Gaius sends) or `raw_input_contents` (what tritonclient sends by default —
        4-byte-LE-length-prefixed framing per tensor)."""
        for i, t in enumerate(request.inputs):
            if t.name != name:
                continue
            if t.contents.bytes_contents:
                return t.contents.bytes_contents[0].decode("utf-8")
            if i < len(request.raw_input_contents):
                raw = request.raw_input_contents[i]
                if len(raw) >= 4:
                    n = int.from_bytes(raw[:4], "little")
                    if 4 + n <= len(raw):
                        return raw[4:4 + n].decode("utf-8")
                return raw.decode("utf-8")
        return ""

    def ServerLive(self, request, context):
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        return opb.ServerLiveResponse(live=True)

    def ServerReady(self, request, context):
        # Always ready: the engine is up; whether a capability is served is the peer's answer at
        # ModelInfer time (FAILED_PRECONDITION when no peer serves it — never a local fallback).
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        return opb.ServerReadyResponse(ready=True)

    def ModelReady(self, request, context):
        # RESIDENCY, not configurability: nothing is resident HERE (Ægir hosts no model), so this
        # is always false — peers plan co-tenancy against the hosting engine, not a forwarder.
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        ready = any(e.capability == request.name and e.healthy for e in self.mgr.status())
        return opb.ModelReadyResponse(ready=ready)

    def ServerMetadata(self, request, context):
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        from importlib.metadata import PackageNotFoundError, version
        try:
            v = version("aegir")
        except PackageNotFoundError:
            v = "0.0.0"
        # Extensions advertise the richer co-registered faces so an OIP peer can upgrade.
        return opb.ServerMetadataResponse(
            name="aegir-engine", version=v,
            extensions=["zndx.engine.v1.Engine", "aegir.engine.AegirEngine"])

    def ModelMetadata(self, request, context):
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model name (= capability) is required")
        # Capabilities are federation vocabulary, not a local registry: the tensor contract is the
        # same for any capability this engine forwards. Residency is ModelReady's question.
        tm = opb.ModelMetadataResponse.TensorMetadata
        return opb.ModelMetadataResponse(
            name=request.name, versions=["v1"], platform="zndx-forward",
            inputs=[tm(name="prompt", datatype="BYTES", shape=[1]),
                    tm(name="system_prompt", datatype="BYTES", shape=[1])],
            outputs=[tm(name="completion", datatype="BYTES", shape=[1]),
                     tm(name="reasoning_content", datatype="BYTES", shape=[1])])

    def ModelInfer(self, request, context):
        from aegir.engine.proto import open_inference_grpc_pb2 as opb
        cap = request.model_name or DEFAULT_CAPABILITY
        prompt = self._bytes_input(request, "prompt")
        if not prompt:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'no "prompt" input tensor provided')
        system_prompt = self._bytes_input(request, "system_prompt")
        p = request.parameters
        max_tokens = int(p["max_tokens"].int64_param) if "max_tokens" in p else 2048
        temperature = float(p["temperature"].double_param) if "temperature" in p else 0.7
        json_schema = p["json_schema"].string_param if "json_schema" in p else ""
        try:
            out = self.mgr.complete(cap, prompt, system_prompt, max_tokens, temperature,
                                    json_schema=json_schema)
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except grpc.RpcError as e:
            context.abort(e.code(), f"infer[{cap}] refused by the serving peer: {e.details()}")
        except Exception as e:  # noqa: BLE001 — surface as a gRPC error, keep the engine up
            context.abort(grpc.StatusCode.INTERNAL, f"infer[{cap}] failed: {e}")
        resp = opb.ModelInferResponse(model_name=cap, model_version="v1", id=request.id)
        o = resp.outputs.add()
        o.name, o.datatype = "completion", "BYTES"
        o.shape.extend([1])
        o.contents.bytes_contents.append(out["text"].encode("utf-8"))
        if out.get("reasoning_content"):
            r = resp.outputs.add()
            r.name, r.datatype = "reasoning_content", "BYTES"
            r.shape.extend([1])
            r.contents.bytes_contents.append(out["reasoning_content"].encode("utf-8"))
        resp.parameters["tokens_used"].int64_param = out["completion_tokens"]
        resp.parameters["prompt_tokens"].int64_param = out["prompt_tokens"]
        resp.parameters["latency_ms"].double_param = out["latency_ms"]
        resp.parameters["backend"].string_param = out.get("fulfilled_by") or "zndx-forward"
        resp.parameters["model"].string_param = out["model"]
        resp.parameters["finish_reason"].string_param = out["finish_reason"]
        return resp


def serve(port: int = ENGINE_GRPC_PORT) -> None:
    # Stamp the commit this process runs from ONCE, before serving: ServerQuery
    # SOURCE_POSTURE reports it beside the live HEAD so peers see a moved checkout.
    from aegir.engine.s2s import stamp_running_sha
    stamp_running_sha()
    servicer = AegirEngineServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pbg.add_AegirEngineServicer_to_server(servicer, server)
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg
    zpbg.add_EngineServicer_to_server(ZndxEngineServicer(servicer.mgr), server)
    from aegir.engine.proto import open_inference_grpc_pb2_grpc as opbg
    opbg.add_GRPCInferenceServiceServicer_to_server(OipInferenceServicer(servicer.mgr), server)
    enable_reflection(server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"aegir-engine gRPC listening on :{port} — services: aegir.engine.AegirEngine + "
          f"zndx.engine.v1.Engine (federation face) + inference.GRPCInferenceService (KServe OIP) "
          f"+ reflection; hosts no model — inference capabilities are forwarded to the federation",
          flush=True)

    def _stop(*_):
        servicer.mgr.shutdown()
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
