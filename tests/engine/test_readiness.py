"""Unit tests for ``aegir.engine.readiness`` — readiness probe / wait.

The probe reaches the engine through ``aegir.engine.client`` (engine_status / complete); both are
monkeypatched, so no gRPC channel, no vLLM, no live ports.
"""
from __future__ import annotations

import types

import pytest

from aegir.engine import client as engine_client
from aegir.engine.readiness import Readiness, probe, wait_for_ready


def _status(endpoints):
    """Build a fake EngineStatusResponse-like object."""
    eps = [types.SimpleNamespace(capability=c, healthy=h) for c, h in endpoints]
    return types.SimpleNamespace(endpoints=eps, total_gpus=len(eps))


@pytest.fixture
def patch_client(monkeypatch):
    """Helper to install fake engine_status / complete behaviors."""
    state = {"status": None, "status_exc": None, "complete": None, "complete_exc": None,
             "complete_calls": 0}

    def fake_status(timeout=10.0):
        if state["status_exc"]:
            raise state["status_exc"]
        return state["status"]

    def fake_complete(prompt, *, capability="instruct", **kw):
        state["complete_calls"] += 1
        if state["complete_exc"]:
            raise state["complete_exc"]
        return state["complete"] if state["complete"] is not None else "x"

    monkeypatch.setattr(engine_client, "engine_status", fake_status)
    monkeypatch.setattr(engine_client, "complete", fake_complete)
    return state


def test_unreachable_when_status_raises(patch_client):
    patch_client["status_exc"] = RuntimeError("connection refused")
    level, detail = probe(want_serving=True)
    assert level == Readiness.UNREACHABLE
    assert "EngineStatus failed" in detail


def test_connected_but_not_healthy_without_serving(patch_client):
    patch_client["status"] = _status([("instruct", False)])
    level, detail = probe(want_serving=False)
    assert level == Readiness.CONNECTED
    assert "not yet healthy" in detail


def test_endpoint_healthy_without_serving(patch_client):
    patch_client["status"] = _status([("instruct", True)])
    level, _ = probe(want_serving=False)
    assert level == Readiness.ENDPOINT_HEALTHY


def test_serving_when_complete_succeeds(patch_client):
    patch_client["status"] = _status([("instruct", True)])
    patch_client["complete"] = "ok"
    level, detail = probe(want_serving=True)
    assert level == Readiness.SERVING
    assert patch_client["complete_calls"] == 1


def test_serving_probe_drives_cold_load(patch_client):
    # Endpoint not yet healthy but a Complete (cold load) succeeds → still reaches SERVING.
    patch_client["status"] = _status([])  # no endpoints registered yet
    patch_client["complete"] = "loaded"
    level, _ = probe(want_serving=True)
    assert level == Readiness.SERVING


def test_complete_failure_reports_endpoint_healthy_floor(patch_client):
    # Healthy endpoint but the Complete probe fails → report ENDPOINT_HEALTHY (not SERVING), with reason.
    patch_client["status"] = _status([("instruct", True)])
    patch_client["complete_exc"] = RuntimeError("vLLM 500")
    level, detail = probe(want_serving=True)
    assert level == Readiness.ENDPOINT_HEALTHY
    assert "Complete probe failed" in detail


def test_wait_for_ready_returns_on_first_success(patch_client):
    patch_client["status"] = _status([("instruct", True)])
    patch_client["complete"] = "ok"
    rep = wait_for_ready(level=Readiness.SERVING, timeout=10, poll_interval=0.01)
    assert rep.ok and rep.level == Readiness.SERVING


def test_wait_for_ready_polls_until_ready(patch_client, monkeypatch):
    # First two probes UNREACHABLE, third SERVING. Patch sleep to no-op so the test is instant.
    seq = iter([RuntimeError("x"), RuntimeError("x"), None])

    def fake_status(timeout=10.0):
        nxt = next(seq, None)
        if isinstance(nxt, Exception):
            raise nxt
        return _status([("instruct", True)])

    monkeypatch.setattr(engine_client, "engine_status", fake_status)
    monkeypatch.setattr(engine_client, "complete", lambda *a, **k: "ok")
    import aegir.engine.readiness as rmod
    monkeypatch.setattr(rmod.time, "sleep", lambda *_: None)

    attempts = []
    rep = wait_for_ready(level=Readiness.SERVING, timeout=10, poll_interval=0.0,
                         on_attempt=lambda n, lvl, d: attempts.append(n))
    assert rep.ok
    assert attempts == [1, 2, 3]


def test_wait_for_ready_times_out(patch_client, monkeypatch):
    patch_client["status_exc"] = RuntimeError("down")
    import aegir.engine.readiness as rmod
    # Fake a clock that jumps past the deadline after the first poll.
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(rmod.time, "time", lambda: next(ticks, 100.0))
    monkeypatch.setattr(rmod.time, "sleep", lambda *_: None)
    rep = wait_for_ready(level=Readiness.SERVING, timeout=10, poll_interval=0.0)
    assert not rep.ok
    assert "timed out" in rep.detail
