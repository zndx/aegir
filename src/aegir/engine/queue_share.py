"""Request YK queue guarantee floors over zndx.scheduler.v1.

WRK (instruct vLLM) occupies a resource-class leaf. Guarantees must move over
time so YK preemption can fire. Signals records RequestQueueShare; applying
queues.yaml is Signals later. UNIMPLEMENTED means Signals is behind the proto
— admit still proceeds. QueueHint (Engine/ServerQuery QUEUES) stays the
declared leaf shape, not occupancy-over-time.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger("aegir.engine.queue_share")

GURU_SHAREFAIL = "#YK.00000007.SHAREFAIL"
GPU_KEY = "federation.zndx.org/gpu"
PEER = "aegir"


@dataclass(frozen=True)
class ResourceClass:
    """Leaf resource class. ``name`` is the stamp; ``queue`` is the YK path."""

    name: str
    queue: str
    gpu_tokens: int
    max_applications: int = 1


# Default instruct is TP=4 (Qwen3.6-35B-A3B-FP8) — same leaf as Gaius HEAVY.
HEAVY = ResourceClass(
    name="internal.inference.heavy",
    queue="root.internal.inference.heavy",
    gpu_tokens=4,
    max_applications=1,
)
MEDIUM = ResourceClass(
    name="internal.inference.medium",
    queue="root.internal.inference.medium",
    gpu_tokens=2,
    max_applications=1,
)
LIGHT = ResourceClass(
    name="internal.inference.light",
    queue="root.internal.inference.light",
    gpu_tokens=1,
    max_applications=2,
)


def resource_class_for_capability(capability: str, tp: int | None = None) -> ResourceClass:
    """Map an Aegir capability + tensor-parallel width to a declared leaf."""
    if tp is None:
        from aegir.engine.config import CAPABILITY_MODELS
        spec = CAPABILITY_MODELS.get(capability)
        tp = int(spec.tensor_parallel_size) if spec is not None else 4
    if tp >= 4:
        return HEAVY
    if tp >= 2:
        return MEDIUM
    if tp >= 1:
        return LIGHT
    return HEAVY


def _request_id() -> str:
    gen = getattr(uuid, "uuid7", None)
    return str(gen() if callable(gen) else uuid.uuid4())


def _addr() -> str:
    return (
        os.environ.get("SIGNALS_ENGINE_GRPC")
        or os.environ.get("SIGNALS_ENGINE_TARGET")
        or "127.0.0.1:50551"
    )


def _sharefail(detail: str) -> RuntimeError:
    return RuntimeError(
        f"{GURU_SHAREFAIL} {detail}\n"
        "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
    )


def share_for_class(kind: str, rc: ResourceClass) -> object:
    """Build a QueueShareRequest for one WRK occupying ``rc``."""
    from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2 as spb

    gpu = int(rc.gpu_tokens)
    max_gpu = gpu if gpu >= 2 else (2 if gpu else 0)
    share = spb.QueueShare(
        queue=rc.queue,
        guaranteed=spb.ResourceMap(quantities={GPU_KEY: gpu}),
        max=spb.ResourceMap(quantities={GPU_KEY: max_gpu}),
        max_applications=int(rc.max_applications),
    )
    wrk = spb.WorkloadIntent(
        wrk=kind.replace("_", "-"),
        queue=rc.queue,
        resource_class=rc.name,
        applications=1,
    )
    return spb.QueueShareRequest(
        peer=PEER,
        request_id=_request_id(),
        valid_from_ns=time.time_ns(),
        reason=f"{kind} occupies {rc.queue} (guarantee gpu={gpu})",
        workloads=[wrk],
        shares=[share],
    )


def request_queue_share(kind: str, rc: ResourceClass) -> bool:
    """Tell Signals the occupancy intent. True if recorded.

    ``UNIMPLEMENTED``: proto is ahead of Signals — log, do not fail admit.
    Any other gRPC error: fail-fast SHAREFAIL.
    """
    import grpc

    from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = share_for_class(kind, rc)
    channel = grpc.insecure_channel(_addr())
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.RequestQueueShare(req, timeout=5.0)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.info(
                "RequestQueueShare UNIMPLEMENTED (Signals behind proto) wrk=%s queue=%s",
                kind,
                rc.queue,
            )
            return False
        raise _sharefail(f"RequestQueueShare failed: {e.code()} {e.details()}") from e
    finally:
        channel.close()
    if not resp.accepted and (resp.error or "").strip():
        raise _sharefail(resp.error)
    log.info(
        "RequestQueueShare accepted=%s state=%s wrk=%s queue=%s id=%s",
        resp.accepted,
        resp.state,
        kind,
        rc.queue,
        resp.request_id or req.request_id,
    )
    return bool(resp.accepted)


def list_queue_share_requests(
    *,
    peer: str = "",
    queue: str = "",
    since_ns: int = 0,
    limit: int = 0,
) -> list:
    """List occupancy-intent history from Signals. Empty on UNIMPLEMENTED."""
    import grpc

    from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2 as spb
    from aegir.engine.proto.zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = spb.ListQueueShareRequestsRequest(
        peer=peer or PEER, queue=queue, since_ns=since_ns, limit=limit)
    channel = grpc.insecure_channel(_addr())
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.ListQueueShareRequests(req, timeout=5.0)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.info("ListQueueShareRequests UNIMPLEMENTED (Signals behind proto)")
            return []
        raise _sharefail(
            f"ListQueueShareRequests failed: {e.code()} {e.details()}"
        ) from e
    finally:
        channel.close()
    return list(resp.records)
