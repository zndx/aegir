"""Capability forwarder — Ægir hosts no model; inference is served by the peer that hosts it.

Strict layering is preserved on both sides: workloads still speak only to THIS engine (``client``),
and this engine speaks only ``zndx.engine.v1`` to peers — never a vLLM port, never an OpenAI URL.

Resolution (signals-protocol ``capabilities.md`` §Operating profiles): a peer SERVES a capability
iff its ``Engine/Status`` lists an ``Endpoint`` with that capability **healthy**. Peers are the
configured lattice (Signals peer-contract) plus live ``Announce``'d peers, self excluded; the first
healthy offer wins and Status answers are cached ``PEER_STATUS_TTL_S``. A forwarding engine never
lists the capability itself, so forwarders cannot chain into a loop.

**No fallback.** If no peer advertises the capability the request fails fast with
:class:`NoPeerServes` (``FAILED_PRECONDITION`` at the faces) naming the peers asked. An ``instruct``
request is never quietly served as ``thinking`` here — the SERVING engine owns the operating profile
and reports what it applied (``CompleteResponse.profile``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import grpc

from aegir.engine.config import (
    DEFAULT_CAPABILITY,
    FORWARD_TIMEOUT_S,
    PEER_STATUS_TIMEOUT_S,
    PEER_STATUS_TTL_S,
    ROUTE_CAPABILITIES,
)
from aegir.engine.proto.zndx.engine.v1 import engine_pb2 as zpb
from aegir.engine.proto.zndx.engine.v1 import engine_pb2_grpc as zpbg

log = logging.getLogger("aegir.engine.forwarder")

GURU_NOPEER = "#AE.00000001.NOPEER"


class NoPeerServes(RuntimeError):
    """No federation peer advertises the capability healthy. Not a fallback point — an answer."""

    def __init__(self, capability: str, asked: list[str]) -> None:
        self.capability = capability
        self.asked = list(asked)
        who = ", ".join(asked) if asked else "no peers configured or announced"
        super().__init__(
            f"{GURU_NOPEER} aegir hosts no models and no federation peer advertises capability "
            f"{capability!r} healthy (asked: {who}).\n"
            f"  The peer that hosts the model must list {capability!r} in Engine/Status "
            f"(capabilities.md §Operating profiles); aegir does not fall back to another capability.")


@dataclass(frozen=True)
class Route:
    capability: str
    peer: str
    target: str
    model: str
    gpu_ids: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.peer}@{self.target}"


@dataclass
class _Cached:
    at: float
    status: "zpb.StatusResponse | None"


def _default_peers() -> list[tuple[str, str]]:
    from aegir.engine.s2s import PROJECT, announced_peers, configured_peers

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for pid, tgt in (*configured_peers(None), *announced_peers()):
        key = (pid or "").strip().lower() or tgt
        if not tgt or key == PROJECT or key in seen:
            continue
        seen.add(key)
        out.append((pid, tgt))
    return out


class CapabilityForwarder:
    """Resolve capability → peer by Status, forward Complete on the zndx face. Hosts nothing.

    Test seams: ``peers`` (callable → [(project, target)]), ``status_fn`` (target → StatusResponse or
    raise), ``forward_fn`` (target, request, timeout → CompleteResponse or raise grpc.RpcError).
    """

    def __init__(self, *, peers=None, status_fn=None, forward_fn=None,
                 ttl_s: float = PEER_STATUS_TTL_S, status_timeout_s: float = PEER_STATUS_TIMEOUT_S,
                 timeout_s: float = FORWARD_TIMEOUT_S, clock=time.monotonic) -> None:
        self._peers = peers or _default_peers
        self._status_fn = status_fn or self._status_over_grpc
        self._forward_fn = forward_fn or self._forward_over_grpc
        self._ttl = ttl_s
        self._status_timeout = status_timeout_s
        self._timeout = timeout_s
        self._clock = clock
        self._cache: dict[str, _Cached] = {}
        self._channels: dict[str, grpc.Channel] = {}
        self._lock = threading.Lock()

    # ── transport ────────────────────────────────────────────────────────────────
    def _channel(self, target: str) -> grpc.Channel:
        with self._lock:
            ch = self._channels.get(target)
            if ch is None:
                ch = grpc.insecure_channel(target)
                self._channels[target] = ch
            return ch

    def _status_over_grpc(self, target: str) -> zpb.StatusResponse:
        return zpbg.EngineStub(self._channel(target)).Status(
            zpb.StatusRequest(), timeout=self._status_timeout)

    def _forward_over_grpc(self, target: str, request: zpb.CompleteRequest,
                           timeout: float) -> zpb.CompleteResponse:
        return zpbg.EngineStub(self._channel(target)).Complete(request, timeout=timeout)

    # ── resolution ───────────────────────────────────────────────────────────────
    def _status(self, target: str) -> "zpb.StatusResponse | None":
        now = self._clock()
        c = self._cache.get(target)
        if c is not None and now - c.at < self._ttl:
            return c.status
        try:
            st = self._status_fn(target)
        except Exception as e:  # noqa: BLE001 — an unreachable peer is "offers nothing" for one TTL
            log.info("peer status failed target=%s err=%s", target, type(e).__name__)
            st = None
        self._cache[target] = _Cached(at=now, status=st)
        return st

    def invalidate(self, target: "str | None" = None) -> None:
        if target is None:
            self._cache.clear()
        else:
            self._cache.pop(target, None)

    def resolve(self, capability: str) -> Route:
        cap = (capability or DEFAULT_CAPABILITY).strip()
        asked: list[str] = []
        for pid, tgt in self._peers():
            st = self._status(tgt)
            asked.append(f"{pid or '?'}@{tgt}")
            if st is None:
                continue
            for ep in st.endpoints:
                if ep.capability == cap and ep.healthy:
                    return Route(capability=cap, peer=st.project or pid, target=tgt,
                                 model=ep.model, gpu_ids=tuple(ep.gpu_ids))
        raise NoPeerServes(cap, asked)

    def routes(self, capabilities: tuple[str, ...] = ROUTE_CAPABILITIES) -> list[Route]:
        """Resolved routes only — a capability nobody serves is simply absent (honest)."""
        out: list[Route] = []
        for cap in capabilities:
            try:
                out.append(self.resolve(cap))
            except NoPeerServes:
                continue
        return out

    # ── forwarding ───────────────────────────────────────────────────────────────
    def forward(self, request: zpb.CompleteRequest, *, timeout: "float | None" = None) -> zpb.CompleteResponse:
        """Forward a zndx Complete VERBATIM (tools_json / messages_json / capabilities[] included)."""
        if not request.capability and not list(request.capabilities):
            request.capability = DEFAULT_CAPABILITY
        cap = request.capability or DEFAULT_CAPABILITY
        route = self.resolve(cap)
        t0 = time.time()
        try:
            resp = self._forward_fn(route.target, request, timeout or self._timeout)
        except grpc.RpcError as e:
            # A peer that was healthy a moment ago and now refuses is stale cache — drop it so the
            # next call re-resolves. The error itself is the caller's answer (no retry, no fallback).
            self.invalidate(route.target)
            raise
        log.info("forwarded capability=%s → %s model=%s in %.0fms",
                 cap, route.label, resp.model, (time.time() - t0) * 1000.0)
        if not resp.fulfilled_by:
            resp.fulfilled_by = f"forwarded@{route.label}"
        return resp

    def complete(self, capability: str, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.7, json_schema: str = "") -> dict:
        """The dict form the native and OIP faces speak (same keys as the retired local manager)."""
        req = zpb.CompleteRequest(capability=capability or DEFAULT_CAPABILITY, prompt=prompt,
                                  system_prompt=system_prompt or "", max_tokens=int(max_tokens or 0),
                                  temperature=float(temperature or 0.0), json_schema=json_schema or "")
        t0 = time.time()
        r = self.forward(req)
        return {"text": r.text, "model": r.model, "reasoning_content": r.reasoning_content,
                "finish_reason": r.finish_reason, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_ms": r.latency_ms or (time.time() - t0) * 1000.0,
                "fulfilled_by": r.fulfilled_by,
                "profile": {"capability": r.profile.capability, "thinking": r.profile.thinking,
                            "reasoning_effort": r.profile.reasoning_effort}
                if r.HasField("profile") else None}

    # ── the manager surface the faces already use ───────────────────────────────
    def status(self) -> list:
        """HOSTED endpoints: none. (Routes are the native EngineStatus, not the lattice Status.)"""
        return []

    def shutdown(self) -> None:
        with self._lock:
            for ch in self._channels.values():
                try:
                    ch.close()
                except Exception:  # noqa: BLE001
                    pass
            self._channels.clear()
