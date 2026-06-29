"""Unit tests for ``aegir.engine.supervisor`` — supervised restart with backoff.

Fully isolated: a fake "process" is injected via the ``spawn`` seam, time is faked via ``sleep``/``now``,
the GPU guard and readiness check are injected. NO subprocess, NO GPU, NO real sleeping, NO live ports.
"""
from __future__ import annotations

import signal

import pytest

from aegir.engine.supervisor import EngineSupervisor, SupervisorConfig


class FakeProc:
    """A scriptable stand-in for subprocess.Popen.

    ``exit_after`` poll() calls it stays alive, then reports ``returncode``. ``-1`` => never exit on its own
    (the test will stop the supervisor instead).
    """

    _next_pid = 1000

    def __init__(self, exit_after: int, returncode: int):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self._exit_after = exit_after
        self.returncode_when_done = returncode
        self.returncode = None
        self._polls = 0
        self.killed_with = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        self._polls += 1
        if self._exit_after >= 0 and self._polls >= self._exit_after:
            self.returncode = self.returncode_when_done
        return self.returncode

    # The supervisor's shutdown path goes through os.killpg; only the fallback uses send_signal.
    def send_signal(self, sig):
        self.killed_with = sig
        self.returncode = -sig


class Clock:
    """Deterministic monotonic clock; ``sleep`` advances it. Records total slept for backoff assertions."""

    def __init__(self):
        self.t = 0.0
        self.slept = 0.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.slept += s
        self.t += s


def _events(captured):
    return [e["event"] for e in captured]


def make_sup(procs, *, cfg=None, readiness=None, claim=None, clock=None, gpu_ids=None):
    """Build a supervisor whose spawn yields each FakeProc in ``procs`` in turn."""
    clock = clock or Clock()
    it = iter(procs)
    captured: list[dict] = []

    def spawn():
        return next(it)

    sup = EngineSupervisor(
        cfg or SupervisorConfig(),
        spawn=spawn,
        readiness_check=readiness,
        claim=claim,
        gpu_ids=gpu_ids,
        sleep=clock.sleep,
        now=clock.now,
        log=captured.append,
    )
    # Disable the auto-default readiness/guard wiring so a None stays None (pure injection).
    sup._use_default_readiness = False  # type: ignore[attr-defined]
    sup._use_default_guard = False  # type: ignore[attr-defined]
    return sup, captured, clock


# ---------------------------------------------------------------------------
def test_clean_exit_stops_supervisor():
    sup, cap, _ = make_sup([FakeProc(exit_after=2, returncode=0)])
    rc = sup.run()
    assert rc == 0
    assert "exit_clean" in _events(cap)
    assert "exit_crash" not in _events(cap)


def test_crash_then_clean_restarts_once():
    # First child crashes (rc=1), second exits clean → supervisor restarts once then stops cleanly.
    sup, cap, clock = make_sup([
        FakeProc(exit_after=1, returncode=1),   # crash
        FakeProc(exit_after=1, returncode=0),   # clean
    ])
    rc = sup.run()
    assert rc == 0
    ev = _events(cap)
    assert ev.count("start") == 2
    assert "exit_crash" in ev and "backoff" in ev and "exit_clean" in ev
    # One backoff of base (2s) happened.
    assert clock.slept >= 2.0


def test_exponential_backoff_sequence():
    # Three crashes then clean: backoff delays must be 2, 4, 8 (base=2, doubling).
    cfg = SupervisorConfig(backoff_base_s=2.0, backoff_cap_s=120.0, poll_interval_s=2.0,
                           max_retries=100)
    sup, cap, clock = make_sup([
        FakeProc(1, 1), FakeProc(1, 1), FakeProc(1, 1), FakeProc(1, 0),
    ], cfg=cfg)
    rc = sup.run()
    assert rc == 0
    backoffs = [e["seconds"] for e in cap if e["event"] == "backoff"]
    assert backoffs == [2.0, 4.0, 8.0]


def test_backoff_is_capped():
    sup, _, _ = make_sup([])  # just exercise the helper
    sup.cfg = SupervisorConfig(backoff_base_s=2.0, backoff_cap_s=10.0)
    assert sup._backoff_for(1) == 2.0
    assert sup._backoff_for(10) == 10.0  # 2*2**9 = 1024, capped to 10


def test_giveup_after_max_retries_in_window():
    # max_retries=2 within a huge window: the 3rd crash exceeds the cap → giveup, rc=1.
    cfg = SupervisorConfig(max_retries=2, window_s=1e9, backoff_base_s=1.0, poll_interval_s=1.0)
    procs = [FakeProc(1, 1) for _ in range(5)]  # all crash; supervisor never gets a clean exit
    sup, cap, _ = make_sup(procs, cfg=cfg)
    rc = sup.run()
    assert rc == 1
    assert "giveup" in _events(cap)
    # 3 crashes recorded (1 over the cap of 2), then giveup — not all 5 procs consumed.
    assert _events(cap).count("exit_crash") == 3


def test_rolling_window_evicts_old_restarts():
    # With a SHORT window, crashes spaced far apart don't accumulate → never gives up.
    # backoff base huge so the clock advances past the window between crashes.
    cfg = SupervisorConfig(max_retries=1, window_s=5.0, backoff_base_s=100.0, backoff_cap_s=100.0,
                           poll_interval_s=100.0)
    sup, cap, _ = make_sup([
        FakeProc(1, 1), FakeProc(1, 1), FakeProc(1, 0),
    ], cfg=cfg)
    rc = sup.run()
    # Each backoff (100s) >> window (5s), so each crash is alone in its window → max_retries=1 never exceeded.
    assert rc == 0
    assert "giveup" not in _events(cap)


def test_signal_killed_child_is_a_crash_not_clean():
    # A child killed by SIGSEGV (rc=-11) must be treated as a crash and restarted.
    sup, cap, _ = make_sup([
        FakeProc(1, -signal.SIGSEGV),
        FakeProc(1, 0),
    ])
    rc = sup.run()
    assert rc == 0
    crash = next(e for e in cap if e["event"] == "exit_crash")
    assert "SIGSEGV" in crash["reason"]


def test_describe_exit_classification():
    assert EngineSupervisor._describe_exit(0) == (True, "clean exit (rc=0)")
    is_clean, reason = EngineSupervisor._describe_exit(-signal.SIGTERM)
    assert not is_clean and "SIGTERM" in reason
    is_clean, reason = EngineSupervisor._describe_exit(1)
    assert not is_clean and "non-zero" in reason
    assert EngineSupervisor._describe_exit(None)[0] is False


def test_readiness_check_invoked_after_start():
    calls = []

    def readiness():
        calls.append(1)
        return True, "serving (probe ok)"

    sup, cap, _ = make_sup([FakeProc(2, 0)], readiness=readiness)
    sup.run()
    assert len(calls) == 1
    rd = next(e for e in cap if e["event"] == "readiness")
    assert rd["ok"] is True


def test_gpu_guard_claimed_and_released():
    entered, exited = [], []

    class FakeGuard:
        def __init__(self, gpus):
            self.gpus = gpus

        def __enter__(self):
            entered.append(self.gpus)
            return self

        def __exit__(self, *exc):
            exited.append(self.gpus)
            return False

    sup, cap, _ = make_sup([FakeProc(2, 0)], claim=FakeGuard, gpu_ids=[0, 1, 2, 3])
    sup.run()
    assert entered == [[0, 1, 2, 3]]
    assert exited == [[0, 1, 2, 3]]
    assert "gpu_claimed" in _events(cap)


def test_gpu_guard_failure_aborts_run():
    from aegir.engine.gpu_guard import GpuClaimError

    def claim(_gpus):
        raise GpuClaimError("GPUs [0,1,2,3] already in use")

    sup, cap, _ = make_sup([FakeProc(2, 0)], claim=claim, gpu_ids=[0, 1, 2, 3])
    with pytest.raises(GpuClaimError):
        sup.run()
    # Never spawned the server because the guard refused first.
    assert "start" not in _events(cap)


def test_request_stop_tears_down_child_cleanly(monkeypatch):
    # A long-running child + a stop request → clean shutdown via killpg path, supervisor returns 0.
    proc = FakeProc(exit_after=-1, returncode=0)  # never exits on its own

    killpg_calls = []
    monkeypatch.setattr("aegir.engine.supervisor.os.killpg",
                        lambda pid, sig: killpg_calls.append((pid, sig)) or setattr(proc, "returncode", -sig))
    monkeypatch.setattr("aegir.engine.supervisor.os.getpgid", lambda pid: pid)

    clock = Clock()
    captured: list[dict] = []
    it = iter([proc])
    sup = EngineSupervisor(SupervisorConfig(poll_interval_s=1.0),
                           spawn=lambda: next(it),
                           readiness_check=lambda: (True, "ok"),
                           sleep=clock.sleep, now=clock.now, log=captured.append)
    sup._use_default_readiness = False  # type: ignore[attr-defined]
    sup._use_default_guard = False  # type: ignore[attr-defined]

    # Simulate the stop arriving after the child started: poll once, then request stop.
    orig_poll = proc.poll
    polls = {"n": 0}

    def poll_then_stop():
        polls["n"] += 1
        if polls["n"] == 2 and not sup._stopping:
            sup.request_stop()
        return orig_poll()

    proc.poll = poll_then_stop
    rc = sup.run()
    assert rc == 0
    assert killpg_calls, "killpg must be used to tear down the child group (no orphaned VRAM)"
    ev = _events(captured)
    assert "stop_requested" in ev and "supervisor_stopped" in ev


def test_shutdown_child_uses_killpg(monkeypatch):
    proc = FakeProc(exit_after=-1, returncode=0)
    killpg = []
    monkeypatch.setattr("aegir.engine.supervisor.os.killpg",
                        lambda pid, sig: killpg.append((pid, sig)) or setattr(proc, "returncode", -sig))
    monkeypatch.setattr("aegir.engine.supervisor.os.getpgid", lambda pid: pid)

    clock = Clock()
    sup = EngineSupervisor(SupervisorConfig(), spawn=lambda: proc,
                           sleep=clock.sleep, now=clock.now, log=lambda e: None)
    sup._proc = proc
    sup.shutdown_child(signal.SIGTERM)
    assert killpg and killpg[0][1] == signal.SIGTERM
