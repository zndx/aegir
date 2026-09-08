"""Tests for the zndx.engine.v1 lattice face: Status body + server reflection.

Ægir hosts NO model (2026-09-08): Status advertises project=aegir with EMPTY endpoints — an engine
that forwards a capability must not claim it (capabilities.md §Operating profiles). Lattice accept is
Engine/Status answering at gRPC bind; reflection is the external grpcurl surface.
"""
from __future__ import annotations

from concurrent import futures
from types import SimpleNamespace

import grpc

from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg
from aegir.engine.server import PROJECT, ZndxEngineServicer, build_status_response, enable_reflection


def _mgr(endpoints=()):
    return SimpleNamespace(status=lambda: list(endpoints))


class TestBuildStatusResponse:
    def test_advertises_aegir_with_no_hosted_endpoints(self):
        resp = build_status_response(_mgr())
        assert resp.project == PROJECT
        assert list(resp.endpoints) == []  # nothing hosted — honest, never a placeholder

    def test_never_claims_instruct_or_thinking(self):
        resp = build_status_response(_mgr())
        assert not any(ep.capability in ("instruct", "thinking") for ep in resp.endpoints)

    def test_reports_total_gpus_and_surfaces_fields(self):
        resp = build_status_response(_mgr())
        assert resp.total_gpus >= 0
        assert hasattr(resp, "surfaces")


class TestZndxServicerStatus:
    def test_status_rpc(self):
        servicer = ZndxEngineServicer(_mgr())
        resp = servicer.Status(zpb.StatusRequest(), None)
        assert resp.project == "aegir"
        assert list(resp.endpoints) == []


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
            assert list(r.endpoints) == []
            ch.close()
        finally:
            server.stop(grace=0)
