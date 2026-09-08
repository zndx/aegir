"""Readiness probe — confirm the engine is *actually serving* before a long workload commits.

Ægir hosts no model: ``EngineStatus`` (native face) lists the federation ROUTES the engine resolved
(peer whose Status serves the capability), and the first ``Complete`` is forwarded (formerly vLLM launched and
health-waits up to 900 s on that call). So "the gRPC port is open" is NOT the same as "ready": a workload
that fires its first real request at a cold engine either blocks for ~minutes on the cold load or, if it
has a short deadline, times out and fails. For a multi-day derivation run we want the *driver* to WAIT for
genuine readiness up front, then proceed knowing every subsequent ``Complete`` is warm.

Three escalating levels (:class:`Readiness`):

* ``CONNECTED`` — the gRPC channel is up and ``EngineStatus`` answers (server process alive).
* ``ENDPOINT_HEALTHY`` — ``EngineStatus``/``EnsureEndpoint`` reports the capability's vLLM endpoint
  ``healthy`` (model loaded into VRAM).
* ``SERVING`` — a trivial ``Complete`` (1 token) round-trips successfully (the full path works end-to-end).

:func:`wait_for_ready` polls to a target level with a deadline, returning a :class:`ReadinessReport`. Use
it at the top of a long run::

    from aegir.engine.readiness import wait_for_ready, Readiness
    rep = wait_for_ready(level=Readiness.SERVING, timeout=1200)
    if not rep.ok:
        raise SystemExit(f"engine not ready: {rep.detail}")
    # ... now fan out the derivation workload; every Complete is warm.

Test seam: the probe talks to the engine through the public ``aegir.engine.client`` functions
(``engine_status`` / ``complete``), which are monkeypatched in the unit tests — no live gRPC needed.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass


class Readiness(enum.IntEnum):
    """Escalating readiness levels (ordered: a higher level implies all lower ones hold)."""

    UNREACHABLE = 0      # cannot even open the channel / EngineStatus failed
    CONNECTED = 1        # server process answers gRPC
    ENDPOINT_HEALTHY = 2  # capability's vLLM endpoint reports healthy (model loaded)
    SERVING = 3          # a trivial Complete round-tripped


@dataclass
class ReadinessReport:
    level: Readiness
    target: Readiness
    detail: str
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.level >= self.target


def probe(capability: str = "instruct", *, want_serving: bool = True,
          status_timeout: float = 10.0, complete_timeout: float = 900.0) -> "tuple[Readiness, str]":
    """One-shot probe. Returns ``(level_reached, detail)``. Never raises — failures map to a level + reason.

    ``want_serving=False`` stops at ``ENDPOINT_HEALTHY`` (skips the trivial ``Complete``) — useful for a
    cheap liveness check that must not itself trigger a cold model load. With ``want_serving=True`` (the
    default), reaching ``ENDPOINT_HEALTHY`` but not yet ``SERVING`` still issues the ``Complete``; on a cold
    engine that Complete is exactly what drives the lazy load, so a SERVING probe doubles as a warm-up.
    """
    # Lazy import so this module (and its tests) don't require a live channel at import time.
    from aegir.engine import client

    # Level 1: CONNECTED — EngineStatus answers.
    try:
        st = client.engine_status(timeout=status_timeout)
    except Exception as e:  # noqa: BLE001 — any transport/timeout error => unreachable
        return Readiness.UNREACHABLE, f"EngineStatus failed: {type(e).__name__}: {e}"

    # Level 2: ENDPOINT_HEALTHY — the named capability's endpoint reports healthy.
    healthy = any(getattr(ep, "capability", None) == capability and getattr(ep, "healthy", False)
                  for ep in getattr(st, "endpoints", []))
    if not healthy:
        if not want_serving:
            return Readiness.CONNECTED, (f"connected ({len(getattr(st, 'endpoints', []))} endpoint(s)) "
                                         f"but {capability!r} not yet healthy")
        # Fall through: a Complete will lazy-load it.
    else:
        if not want_serving:
            return Readiness.ENDPOINT_HEALTHY, f"{capability!r} endpoint healthy"

    # Level 3: SERVING — a trivial Complete works end to end (also performs the cold load if needed).
    try:
        txt = client.complete("ok", capability=capability, max_tokens=1, temperature=0.0,
                              timeout=complete_timeout)
    except Exception as e:  # noqa: BLE001
        # We at least connected; distinguish "healthy-but-Complete-failed" from "still loading".
        base = Readiness.ENDPOINT_HEALTHY if healthy else Readiness.CONNECTED
        return base, f"Complete probe failed: {type(e).__name__}: {e}"
    return Readiness.SERVING, f"serving ({capability!r}); trivial Complete returned {len(txt)} chars"


def wait_for_ready(capability: str = "instruct", *, level: Readiness = Readiness.SERVING,
                   timeout: float = 1200.0, poll_interval: float = 5.0,
                   status_timeout: float = 10.0, complete_timeout: float = 900.0,
                   on_attempt=None) -> ReadinessReport:
    """Poll :func:`probe` until ``level`` is reached or ``timeout`` elapses.

    Returns a :class:`ReadinessReport`; check ``.ok``. Designed to be called once at the start of a long
    workload so the run waits out a cold/loading engine instead of failing against it. ``on_attempt`` (if
    given) is called as ``on_attempt(attempt_idx, reached_level, detail)`` for progress logging.

    Note the single-attempt budget: ``complete_timeout`` defaults to the engine's own 900 s health-wait, so
    ONE SERVING probe can itself ride out a full cold load. ``timeout`` should comfortably exceed it for the
    SERVING level (default 1200 s) to leave room for a retry if the first Complete races the load window.
    """
    want_serving = level >= Readiness.SERVING
    t0 = time.time()
    attempt = 0
    last = Readiness.UNREACHABLE
    detail = "no probe attempted"
    while True:
        attempt += 1
        last, detail = probe(capability, want_serving=want_serving,
                             status_timeout=status_timeout, complete_timeout=complete_timeout)
        if on_attempt is not None:
            with _suppress():
                on_attempt(attempt, last, detail)
        if last >= level:
            return ReadinessReport(level=last, target=level, detail=detail, elapsed_s=time.time() - t0)
        if time.time() - t0 >= timeout:
            return ReadinessReport(level=last, target=level,
                                   detail=f"timed out after {timeout:.0f}s; last: {detail}",
                                   elapsed_s=time.time() - t0)
        # Don't oversleep past the deadline.
        remaining = timeout - (time.time() - t0)
        time.sleep(min(poll_interval, max(0.0, remaining)))


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib for one use site."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
