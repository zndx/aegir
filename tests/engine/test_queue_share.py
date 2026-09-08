"""WRK occupancy intent for Scheduler/RequestQueueShare."""
from __future__ import annotations

import uuid
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest

from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2 as spb
from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2_grpc as spbg
from aegir.engine.queue_share import (
    GPU_KEY,
    GURU_SHAREFAIL,
    HEAVY,
    LIGHT,
    MEDIUM,
    PEER,
    _ADMIT_IDS,
    leaf_for_gpu_tokens,
    list_queue_share_requests,
    mint_uuid7,
    notify_admit,
    notify_release,
    request_queue_share,
    request_queue_share_end,
    require_uuid7,
    resource_class_for_capability,
    share_for_class,
)


@pytest.fixture(autouse=True)
def _clear_admit_ids() -> None:
    _ADMIT_IDS.clear()
    yield
    _ADMIT_IDS.clear()


def test_uuidv7_required_never_v4() -> None:
    s = mint_uuid7()
    u = uuid.UUID(s)
    assert u.version == 7
    assert require_uuid7(s) == s
    with pytest.raises(RuntimeError, match=r"UUIDv7"):
        require_uuid7("")
    with pytest.raises(RuntimeError, match=r"not v4"):
        require_uuid7(str(uuid.uuid4()))


def test_share_mints_uuidv7_every_request() -> None:
    a = share_for_class("instruct", HEAVY)
    b = share_for_class("instruct", HEAVY)
    assert uuid.UUID(a.request_id).version == 7
    assert uuid.UUID(b.request_id).version == 7
    assert a.request_id != b.request_id
    assert a.request_id  # never omit


def test_instruct_default_is_heavy() -> None:
    req = share_for_class("instruct", HEAVY)
    assert req.peer == PEER
    assert req.workloads[0].wrk == "instruct"
    assert req.workloads[0].queue == HEAVY.queue
    assert req.shares[0].queue == HEAVY.queue
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 4
    assert req.shares[0].max.quantities[GPU_KEY] == 4
    assert req.valid_until_ns == 0
    assert uuid.UUID(req.request_id).version == 7


def test_leaf_from_gpu_tokens_not_queue_name() -> None:
    assert leaf_for_gpu_tokens(4) is HEAVY
    assert leaf_for_gpu_tokens(2) is MEDIUM
    assert leaf_for_gpu_tokens(1) is LIGHT
    assert resource_class_for_capability("instruct", 4) is HEAVY
    assert resource_class_for_capability("instruct", 2, 1) is MEDIUM
    assert resource_class_for_capability("instruct", 2, 2) is HEAVY  # tp×pp
    assert resource_class_for_capability("instruct", 1) is LIGHT
    req = share_for_class("instruct", leaf_for_gpu_tokens(4))
    assert req.shares[0].queue == HEAVY.queue
    assert "qwen" not in req.shares[0].queue.lower()
    assert "tp" not in req.shares[0].queue
    assert "pp" not in req.shares[0].queue
    light = share_for_class("instruct", LIGHT)
    assert light.shares[0].guaranteed.quantities[GPU_KEY] == 1
    assert light.shares[0].max.quantities[GPU_KEY] == 2
    assert uuid.UUID(light.request_id).version == 7


def test_zero_floor_valid_until_on_end() -> None:
    req = share_for_class(
        "instruct", HEAVY, gpu=0, valid_until_ns=123, supersedes_request_id="x", applications=0
    )
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 0
    assert req.valid_until_ns == 123
    assert req.workloads[0].applications == 0
    assert req.supersedes_request_id == "x"
    assert uuid.UUID(req.request_id).version == 7
    assert "zero floor" in req.reason


def test_module_never_writes_queues_yaml() -> None:
    from pathlib import Path

    src = Path("src/aegir/engine/queue_share.py").read_text()
    assert "do not write queues.yaml" in src
    assert "open(" not in src
    assert "write_text" not in src
    mgr = Path("src/aegir/engine/vllm_manager.py").read_text()
    assert "notify_admit" in mgr
    assert "notify_release" in mgr
    assert "queues.yaml" not in mgr or "Never writes queues.yaml" in mgr


def _serve(servicer) -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    spbg.add_SchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def test_unimplemented_does_not_fail_admit(monkeypatch) -> None:
    server, addr = _serve(spbg.SchedulerServicer())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert request_queue_share("instruct", HEAVY) is False
        assert list_queue_share_requests() == []
    finally:
        server.stop(grace=0)


class _Reject(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        return spb.QueueShareResponse(
            accepted=False,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_REJECTED,
            error="over parent max",
        )


def test_rejected_does_not_admit(monkeypatch) -> None:
    server, addr = _serve(_Reject())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL") as ei:
            request_queue_share("instruct", HEAVY)
        assert "queues.yaml" in str(ei.value)
        with pytest.raises(RuntimeError, match="SHAREFAIL"):
            notify_admit("instruct")
    finally:
        server.stop(grace=0)


class _RejectSilent(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        return spb.QueueShareResponse(
            accepted=False,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_REJECTED,
        )


def test_rejected_without_error_still_blocks_admit(monkeypatch) -> None:
    server, addr = _serve(_RejectSilent())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"REJECTED"):
            notify_admit("instruct")
    finally:
        server.stop(grace=0)


class _Unavailable(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        context.abort(grpc.StatusCode.UNAVAILABLE, "engine down")

    def ListQueueShareRequests(self, request, context):
        context.abort(grpc.StatusCode.UNAVAILABLE, "engine down")


def test_other_grpc_error_is_sharefail(monkeypatch) -> None:
    server, addr = _serve(_Unavailable())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL") as ei:
            request_queue_share("instruct", HEAVY)
        assert "queues.yaml" in str(ei.value)
        assert GURU_SHAREFAIL in str(ei.value)
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL"):
            list_queue_share_requests()
    finally:
        server.stop(grace=0)


class _Record(spbg.SchedulerServicer):
    def __init__(self) -> None:
        self.seen = []

    def RequestQueueShare(self, request, context):
        self.seen.append(request)
        return spb.QueueShareResponse(
            accepted=True,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_RECORDED,
        )

    def ListQueueShareRequests(self, request, context):
        recs = [
            spb.QueueShareRecord(request=r, recorded_at_ns=1, state=spb.QUEUE_SHARE_RECORDED)
            for r in self.seen
        ]
        return spb.ListQueueShareRequestsResponse(records=recs)


def test_admit_then_zero_floor_end(monkeypatch) -> None:
    svc = _Record()
    server, addr = _serve(svc)
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert notify_admit("instruct") is True
        assert notify_release("instruct") is True
        assert len(svc.seen) == 2
        admit, end = svc.seen
        assert uuid.UUID(admit.request_id).version == 7
        assert uuid.UUID(end.request_id).version == 7
        assert admit.request_id != end.request_id
        assert end.supersedes_request_id == admit.request_id
        assert end.shares[0].guaranteed.quantities[GPU_KEY] == 0
        assert end.valid_until_ns > 0
        assert end.workloads[0].applications == 0
        recs = list_queue_share_requests()
        assert len(recs) == 2
    finally:
        server.stop(grace=0)


def test_send_rejects_non_v7_request_id(monkeypatch) -> None:
    from aegir.engine.queue_share import _send

    req = share_for_class("instruct", HEAVY)
    req.request_id = str(uuid.uuid4())
    with pytest.raises(RuntimeError, match=r"not v4"):
        _send(req)
    req.request_id = ""
    with pytest.raises(RuntimeError, match=r"omitted"):
        _send(req)


def test_notify_admit_unimplemented_ok(monkeypatch) -> None:
    server, addr = _serve(spbg.SchedulerServicer())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert notify_admit("instruct") is False
    finally:
        server.stop(grace=0)


def test_ensure_calls_share_before_launch(monkeypatch) -> None:
    from aegir.engine.vllm_manager import VllmManager

    called = {}

    def fake_admit(kind, tp=None, pp=None):
        called["kind"] = kind
        called["tp"] = tp
        called["pp"] = pp
        return True

    monkeypatch.setattr("aegir.engine.queue_share.notify_admit", fake_admit)
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
    assert called["tp"] == 4
    assert called["pp"] == 1
    assert launched["cap"] == "instruct"


def test_ensure_does_not_launch_on_rejected(monkeypatch) -> None:
    from aegir.engine.vllm_manager import VllmManager

    def boom(kind, tp=None, pp=None):
        raise RuntimeError(f"{GURU_SHAREFAIL} REJECTED")

    monkeypatch.setattr("aegir.engine.queue_share.notify_admit", boom)
    mgr = VllmManager()
    launched: list[str] = []
    monkeypatch.setattr(mgr, "_launch", lambda cap: launched.append(cap))
    monkeypatch.setattr(mgr, "_wait_healthy", lambda ep: None)
    with pytest.raises(RuntimeError, match="SHAREFAIL"):
        mgr.ensure("instruct")
    assert launched == []


def test_shutdown_sends_zero_floor(monkeypatch) -> None:
    from aegir.engine.vllm_manager import VllmManager

    released = []

    def fake_release(kind, tp=None, pp=None):
        released.append((kind, tp, pp))
        return True

    monkeypatch.setattr("aegir.engine.queue_share.notify_release", fake_release)
    mgr = VllmManager()
    mgr._ep["instruct"] = SimpleNamespace(
        capability="instruct",
        spec=SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1),
        proc=None,
    )
    mgr.shutdown()
    assert released == [("instruct", 4, 1)]


def test_declared_queues_are_leaf_shape_not_occupancy() -> None:
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
    from aegir.engine.s2s import declared_queues, local_response

    hints = declared_queues()
    paths = {h.path for h in hints}
    assert "root.internal.inference.heavy" in paths
    for h in hints:
        assert "Qwen" not in h.path
        assert "tp" not in h.path
        assert "pp" not in h.path
        assert "35B" not in h.path
        assert "tp" not in (h.examples or "")
    q = local_response(zpb.SERVER_QUERY_KIND_QUEUES)
    assert [h.path for h in q.queues] == [h.path for h in hints]


def test_workloads_not_in_queue_name() -> None:
    """WORKLOADS carries the typed WorkloadOffer (protocol 7cc9ad1, 2026-08-27):
    model + capabilities + WorkloadRequirements (backend / parallelism / footprint)
    + ResourceClass — never a queue path, never tp/pp folded into a name."""
    from aegir.engine.config import CAPABILITY_MODELS
    from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
    from aegir.engine.s2s import _resource_class_enum, declared_workloads, local_response

    offers = declared_workloads()
    by_cap = {c: o for o in offers for c in o.capabilities}
    instruct = by_cap["instruct"]
    spec = CAPABILITY_MODELS["instruct"]
    tp = spec.tensor_parallel_size
    pp = getattr(spec, "pipeline_parallel_size", 1) or 1
    assert instruct.peer == "aegir"
    assert instruct.model == spec.model
    assert instruct.requirements.backend == zpb.SERVING_BACKEND_VLLM_LOCAL
    assert instruct.requirements.parallelism.tensor_parallel == tp
    assert instruct.requirements.parallelism.pipeline_parallel == pp
    assert instruct.requirements.parallelism.data_parallel == 1
    assert instruct.requirements.footprint.gpu == tp * pp == 4
    assert instruct.resource_class == _resource_class_enum(4)
    assert instruct.resource_class != zpb.RESOURCE_CLASS_UNSPECIFIED
    assert "root.internal" not in instruct.model
    assert instruct.queue == ""  # occupancy is a hint chosen at RequestQueueShare, not baked here
    q = local_response(zpb.SERVER_QUERY_KIND_WORKLOADS)
    assert [(w.peer, w.model, list(w.capabilities)) for w in q.workloads] == [
        (o.peer, o.model, list(o.capabilities)) for o in offers
    ]
    assert list(q.queues) == []
