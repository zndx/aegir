"""Unit tests for ``aegir.engine.gpu_guard`` — GPU-exclusivity guard.

Pure isolation: nvidia-smi is monkeypatched (no GPU touched), FileLock is real but on a tmp dir.
"""
from __future__ import annotations

import json

import pytest

from aegir.engine import gpu_guard
from aegir.engine.gpu_guard import GpuClaimError, claim_gpus, engine_gpu_ids, probe_gpus


@pytest.fixture(autouse=True)
def _tmp_lock_dir(tmp_path, monkeypatch):
    """Point the lock dir at a fresh tmp path for every test so locks/owners never collide or persist."""
    monkeypatch.setattr(gpu_guard, "_LOCK_DIR", tmp_path / "locks")
    return tmp_path / "locks"


def _fake_nvidia(monkeypatch, apps, uuids):
    """Install fake nvidia-smi responses. ``apps`` = [(pid, used_mib, uuid)]; ``uuids`` = {uuid: index}."""
    monkeypatch.setattr(gpu_guard, "_nvidia_compute_apps", lambda: apps)
    monkeypatch.setattr(gpu_guard, "_nvidia_gpu_uuids", lambda: uuids)


def test_engine_gpu_ids_mirrors_config():
    assert engine_gpu_ids(first_gpu=2, tp=3) == [2, 3, 4]
    # Ægir hosts no model (2026-09-08): the default claim is EMPTY — the supervisor skips the guard.
    assert engine_gpu_ids() == []


def test_probe_free_gpus_returns_empty(monkeypatch):
    _fake_nvidia(monkeypatch, apps=[], uuids={"U0": 0, "U1": 1})
    assert probe_gpus([0, 1]) == []


def test_probe_detects_foreign_user(monkeypatch):
    _fake_nvidia(monkeypatch, apps=[(2255740, 21520, "U0"), (2255741, 21520, "U1")],
                 uuids={"U0": 0, "U1": 1, "U4": 4})
    users = probe_gpus([0, 1])
    assert {u.gpu_index for u in users} == {0, 1}
    assert {u.pid for u in users} == {2255740, 2255741}


def test_probe_ignores_other_gpus(monkeypatch):
    # A process on GPU 4 must not register when we only ask about 0-3.
    _fake_nvidia(monkeypatch, apps=[(999, 20000, "U4")], uuids={"U0": 0, "U4": 4})
    assert probe_gpus([0, 1, 2, 3]) == []


def test_probe_respects_min_mib(monkeypatch):
    # Residual context below the floor must not count as "in use".
    _fake_nvidia(monkeypatch, apps=[(123, 100, "U0")], uuids={"U0": 0})
    assert probe_gpus([0], min_mib=512) == []
    assert {u.pid for u in probe_gpus([0], min_mib=50)} == {123}


def test_probe_excludes_self_pids(monkeypatch):
    _fake_nvidia(monkeypatch, apps=[(4242, 20000, "U0")], uuids={"U0": 0})
    assert probe_gpus([0], self_pids={4242}) == []
    assert probe_gpus([0], self_pids={1}) != []


def test_probe_degrades_when_nvidia_unavailable(monkeypatch):
    # If nvidia-smi is missing, the probe returns [] (degrade to lock-only) rather than raising.
    monkeypatch.setattr(gpu_guard, "_nvidia_compute_apps", lambda: None)
    monkeypatch.setattr(gpu_guard, "_nvidia_gpu_uuids", lambda: None)
    assert probe_gpus([0, 1]) == []


def test_claim_free_gpus_succeeds_and_writes_owner(monkeypatch, _tmp_lock_dir):
    _fake_nvidia(monkeypatch, apps=[], uuids={"U0": 0, "U1": 1})
    with claim_gpus([0, 1], role="primary"):
        owner = _tmp_lock_dir / "gpu-0-1.owner.json"
        assert owner.exists()
        d = json.loads(owner.read_text())
        assert d["gpus"] == [0, 1] and d["role"] == "primary"
    # Owner file cleaned up on exit.
    assert not (_tmp_lock_dir / "gpu-0-1.owner.json").exists()


def test_claim_refuses_when_foreign_process_present(monkeypatch):
    _fake_nvidia(monkeypatch, apps=[(2255740, 21520, "U0")], uuids={"U0": 0, "U1": 1})
    with pytest.raises(GpuClaimError) as ei:
        with claim_gpus([0, 1]):
            pass
    assert "foreign compute process" in str(ei.value)
    assert "2255740" in str(ei.value)


def test_claim_releases_lock_on_foreign_refusal(monkeypatch):
    # After a foreign-refusal, the lock must NOT remain held (so a later legit claim can proceed).
    _fake_nvidia(monkeypatch, apps=[(1, 20000, "U0")], uuids={"U0": 0})
    with pytest.raises(GpuClaimError):
        with claim_gpus([0]):
            pass
    # Now the foreign process is gone — a fresh claim must succeed (lock was released).
    _fake_nvidia(monkeypatch, apps=[], uuids={"U0": 0})
    with claim_gpus([0]):
        pass


def test_claim_refuses_second_concurrent_holder(monkeypatch):
    # Two overlapping claims of the same GPU set: the second must fail on the lock (cooperative check),
    # naming the first holder — this is the "two engine instances" guard.
    _fake_nvidia(monkeypatch, apps=[], uuids={"U0": 0, "U1": 1})
    with claim_gpus([0, 1], role="primary"):
        with pytest.raises(GpuClaimError) as ei:
            with claim_gpus([0, 1], role="secondary"):
                pass
        assert "already locked" in str(ei.value)
        assert "role=primary" in str(ei.value)


def test_claim_empty_list_raises():
    with pytest.raises(ValueError):
        with claim_gpus([]):
            pass


def test_nvidia_parse_skips_na_rows(monkeypatch):
    # The real _nvidia_compute_apps must tolerate "[N/A]" memory rows. Drive its parser via a fake _run.
    csv = "2255740, 21520, GPU-aaa\n12345, [N/A], GPU-bbb\n, , \n"
    monkeypatch.setattr(gpu_guard, "_run", lambda *a, **k: csv)
    rows = gpu_guard._nvidia_compute_apps()
    assert (2255740, 21520, "GPU-aaa") in rows
    assert all(r[0] != 12345 for r in rows)  # the [N/A] row was skipped, not crashed on


def test_run_returns_none_on_missing_binary(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpu_guard.subprocess, "run", _boom)
    assert gpu_guard._run(["nvidia-smi", "--foo"]) is None
