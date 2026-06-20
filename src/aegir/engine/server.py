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
                                    request.max_tokens or 512, request.temperature or 0.7)
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


def serve(port: int = ENGINE_GRPC_PORT) -> None:
    servicer = AegirEngineServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pbg.add_AegirEngineServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"aegir-engine gRPC listening on :{port} (capability→model: see engine.config)", flush=True)

    def _stop(*_):
        servicer.mgr.shutdown()
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
