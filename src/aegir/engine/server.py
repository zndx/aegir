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
