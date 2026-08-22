"""Tests for the zndx.engine.v1 lattice face: Status body + server reflection.

Lattice accept is Engine/Status (project=aegir, capability=instruct) at gRPC
bind — no vLLM, no live :50151. Reflection is the external grpcurl surface.
"""
from __future__ import annotations

from concurrent import futures
from types import SimpleNamespace

import grpc

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg
from aegir.engine.server import (
    CAPABILITY_INSTRUCT,
    PROJECT,
    ZndxEngineServicer,
    build_status_response,
    enable_reflection,
)


def _mgr(endpoints=()):
    return SimpleNamespace(status=lambda: list(endpoints))


def _live(capability, model="Qwen/test", healthy=True, gpu_ids=()):
    return SimpleNamespace(
        capability=capability,
        spec=SimpleNamespace(model=model),
        healthy=healthy,
        gpu_ids=list(gpu_ids),
    )


class TestBuildStatusResponse:
    def test_always_advertises_aegir_and_instruct(self):
        resp = build_status_response(_mgr())
        assert resp.project == PROJECT
        caps = [ep.capability for ep in resp.endpoints]
        assert CAPABILITY_INSTRUCT in caps
        instruct = next(ep for ep in resp.endpoints if ep.capability == CAPABILITY_INSTRUCT)
        assert instruct.healthy is True

    def test_overlays_live_instruct_endpoint(self):
        resp = build_status_response(_mgr([
            _live("instruct", model="Qwen/live", healthy=False, gpu_ids=[4, 5]),
        ]))
        instruct = next(ep for ep in resp.endpoints if ep.capability == "instruct")
        assert instruct.model == "Qwen/live"
        assert instruct.healthy is False
        assert list(instruct.gpu_ids) == [4, 5]
        assert sum(1 for ep in resp.endpoints if ep.capability == "instruct") == 1

    def test_appends_other_live_capabilities(self):
        resp = build_status_response(_mgr([_live("reauthor", model="local")]))
        caps = [ep.capability for ep in resp.endpoints]
        assert "instruct" in caps
        assert "reauthor" in caps


class TestZndxServicerStatus:
    def test_status_rpc(self):
        servicer = ZndxEngineServicer(_mgr())
        resp = servicer.Status(zpb.StatusRequest(), None)
        assert resp.project == "aegir"
        assert any(ep.capability == "instruct" for ep in resp.endpoints)


class TestReflection:
    def test_lists_zndx_engine(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        zpbg.add_EngineServicer_to_server(ZndxEngineServicer(_mgr()), server)
        enable_reflection(server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
            ch = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = reflection_pb2_grpc.ServerReflectionStub(ch)
            req = reflection_pb2.ServerReflectionRequest(list_services="")
            replies = list(stub.ServerReflectionInfo(iter([req])))
            names = {s.name for r in replies for s in r.list_services_response.service}
            assert "zndx.engine.v1.Engine" in names
            assert "grpc.reflection.v1alpha.ServerReflection" in names
            r = zpbg.EngineStub(ch).Status(zpb.StatusRequest(), timeout=3)
            assert r.project == "aegir"
            assert any(ep.capability == "instruct" for ep in r.endpoints)
            ch.close()
        finally:
            server.stop(grace=0)
