"""WRK occupancy intent for Scheduler/RequestQueueShare."""
from __future__ import annotations

from types import SimpleNamespace

import grpc
import pytest

from aegir.engine.queue_share import (
    GPU_KEY,
    GURU_SHAREFAIL,
    HEAVY,
    LIGHT,
    MEDIUM,
    PEER,
    list_queue_share_requests,
    request_queue_share,
    resource_class_for_capability,
    share_for_class,
)


def test_instruct_default_is_heavy() -> None:
    req = share_for_class("instruct", HEAVY)
    assert req.peer == PEER
    assert req.workloads[0].wrk == "instruct"
    assert req.workloads[0].queue == HEAVY.queue
    assert req.shares[0].queue == HEAVY.queue
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 4
    assert req.shares[0].max.quantities[GPU_KEY] == 4


def test_tp_maps_to_declared_leaves() -> None:
    assert resource_class_for_capability("instruct", 4) is HEAVY
    assert resource_class_for_capability("instruct", 2) is MEDIUM
    assert resource_class_for_capability("instruct", 1) is LIGHT
    light = share_for_class("instruct", LIGHT)
    assert light.shares[0].guaranteed.quantities[GPU_KEY] == 1
    assert light.shares[0].max.quantities[GPU_KEY] == 2


def test_unimplemented_does_not_fail_admit(monkeypatch) -> None:
    class _Stub:
        def RequestQueueShare(self, req, timeout=5.0):
            err = grpc.RpcError()
            err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
            err.details = lambda: "not implemented"
            raise err

    monkeypatch.setattr("grpc.insecure_channel", lambda addr: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "aegir.engine.proto.zndx.scheduler.v1.scheduler_pb2_grpc.SchedulerStub",
        lambda ch: _Stub(),
    )
    assert request_queue_share("instruct", HEAVY) is False


def test_other_rpc_error_is_sharefail(monkeypatch) -> None:
    class _Stub:
        def RequestQueueShare(self, req, timeout=5.0):
            err = grpc.RpcError()
            err.code = lambda: grpc.StatusCode.UNAVAILABLE
            err.details = lambda: "connection refused"
            raise err

    monkeypatch.setattr("grpc.insecure_channel", lambda addr: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "aegir.engine.proto.zndx.scheduler.v1.scheduler_pb2_grpc.SchedulerStub",
        lambda ch: _Stub(),
    )
    with pytest.raises(RuntimeError, match="YK.00000007.SHAREFAIL") as ei:
        request_queue_share("instruct", HEAVY)
    assert "queues.yaml" in str(ei.value)
    assert GURU_SHAREFAIL in str(ei.value)


def test_rejected_response_is_sharefail(monkeypatch) -> None:
    class _Stub:
        def RequestQueueShare(self, req, timeout=5.0):
            return SimpleNamespace(accepted=False, error="disk full", request_id=req.request_id, state=0)

    monkeypatch.setattr("grpc.insecure_channel", lambda addr: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "aegir.engine.proto.zndx.scheduler.v1.scheduler_pb2_grpc.SchedulerStub",
        lambda ch: _Stub(),
    )
    with pytest.raises(RuntimeError, match="SHAREFAIL"):
        request_queue_share("instruct", HEAVY)


def test_list_unimplemented_returns_empty(monkeypatch) -> None:
    class _Stub:
        def ListQueueShareRequests(self, req, timeout=5.0):
            err = grpc.RpcError()
            err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
            err.details = lambda: "not implemented"
            raise err

    monkeypatch.setattr("grpc.insecure_channel", lambda addr: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "aegir.engine.proto.zndx.scheduler.v1.scheduler_pb2_grpc.SchedulerStub",
        lambda ch: _Stub(),
    )
    assert list_queue_share_requests() == []


def test_ensure_calls_share_before_launch(monkeypatch) -> None:
    from aegir.engine.vllm_manager import VllmManager

    called = {}

    def fake_share(kind, rc):
        called["kind"] = kind
        called["queue"] = rc.queue
        return True

    monkeypatch.setattr("aegir.engine.queue_share.request_queue_share", fake_share)
    mgr = VllmManager()
    launched = {}

    def fake_launch(capability):
        launched["cap"] = capability
        return SimpleNamespace(
            capability=capability, spec=None, port=8100, gpu_ids=[0, 1, 2, 3],
            proc=SimpleNamespace(poll=lambda: None), healthy=True, log_path=None)

    monkeypatch.setattr(mgr, "_launch", fake_launch)
    monkeypatch.setattr(mgr, "_wait_healthy", lambda ep: None)
    mgr.ensure("instruct")
    assert called["kind"] == "instruct"
    assert called["queue"] == HEAVY.queue
    assert launched["cap"] == "instruct"


def test_declared_queues_are_leaf_shape_not_occupancy() -> None:
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
    from aegir.engine.s2s import declared_queues, local_response

    hints = declared_queues()
    paths = {h.path for h in hints}
    assert "root.internal.inference.heavy" in paths
    q = local_response(zpb.SERVER_QUERY_KIND_QUEUES)
    assert [h.path for h in q.queues] == [h.path for h in hints]
