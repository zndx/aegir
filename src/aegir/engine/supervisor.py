"""Supervised engine restart — keep the gRPC engine alive across crashes for a multi-day run.

The engine (``python -m aegir.engine.server``) can die: vLLM OOMs on a pathological prompt, a TP worker
segfaults, the box hiccups. Unsupervised, that ends a derivation run that should have shrugged the blip
off. This supervisor runs the engine server **as a child process** and, when it exits, restarts it with
exponential backoff, a max-retry cap, and structured (JSONL) logging of every restart and its reason.

What it does, in order:

1. **GPU-exclusivity guard** (``gpu_guard.claim_gpus``) up front — refuse to start if the engine's GPU
   range is already claimed by another instance or a foreign process. Fail fast, loud.
2. **Spawn** the server in its OWN process group (``start_new_session=True``), exactly like
   the retired vLLM manager did for vLLM, so we can kill the whole tree.
3. **Readiness wait** after each (re)start (``readiness.wait_for_ready``) — log when the engine is actually
   serving (or that it failed to come up), so the picture is "supervised AND warm", not just "process running".
4. **Detect exit**: when the child exits, log the reason (return code, signal, recent log tail), then:
   * a *clean* exit (rc 0, or killed by our own shutdown signal) ends the supervisor;
   * a *crash* triggers a backoff sleep and a restart, until ``max_retries`` within the rolling window is hit.
5. **Restart storm guard**: retries are counted in a rolling ``window_s``; > ``max_retries`` crashes inside
   one window ⇒ give up (a tight crash loop is a real bug, not a transient — don't thrash GPUs forever).
6. **Clean shutdown preserved**: on SIGINT/SIGTERM to the supervisor we ``killpg`` the child's group
   (mirroring ``server._stop``) so nothing leaks, then exit. (Ægir hosts no model since 2026-09-08.)

The supervisor itself imports NO vLLM and grabs NO GPU — it only spawns the server child (which owns all of
that) and polls it. That keeps it safe to unit-test in isolation: :class:`EngineSupervisor` takes injectable
``spawn`` / ``readiness_check`` / ``sleep`` / ``now`` / ``claim`` seams, so the whole supervise/backoff/giveup
loop is driven by a fake "process" with zero subprocesses, GPUs, or real time.

CLI::

    uv run --no-sync python -m aegir.engine.supervisor              # supervise with defaults
    AEGIR_ENGINE_MAX_RETRIES=10 ... python -m aegir.engine.supervisor --no-gpu-guard

Layering note: this is an OPS wrapper *around* the server, not a new gRPC surface. Workloads still talk only
to ``client.complete``; the supervisor is what a human (or ``just engine-serve``) launches instead of the
bare server when they want resilience.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_LOG_DIR = Path(os.environ.get("AEGIR_ENGINE_LOG_DIR", "/tmp/aegir-engine"))


@dataclass
class SupervisorConfig:
    """Backoff / retry / readiness knobs (env-overridable; CLI overrides env)."""

    max_retries: int = int(os.environ.get("AEGIR_ENGINE_MAX_RETRIES", "8"))
    backoff_base_s: float = float(os.environ.get("AEGIR_ENGINE_BACKOFF_BASE", "2.0"))
    backoff_cap_s: float = float(os.environ.get("AEGIR_ENGINE_BACKOFF_CAP", "120.0"))
    window_s: float = float(os.environ.get("AEGIR_ENGINE_RETRY_WINDOW", "1800.0"))
    poll_interval_s: float = float(os.environ.get("AEGIR_ENGINE_POLL", "2.0"))
    readiness_timeout_s: float = float(os.environ.get("AEGIR_ENGINE_READY_TIMEOUT", "1200.0"))
    gpu_guard: bool = os.environ.get("AEGIR_ENGINE_GPU_GUARD", "1") != "0"
    capability: str = os.environ.get("AEGIR_ENGINE_CAPABILITY", "instruct")


def _default_spawn(log_path: Path) -> subprocess.Popen:
    """Spawn ``python -m aegir.engine.server`` in its own session, mirroring the live-run launch.

    Uses the SAME interpreter that is running the supervisor (``sys.executable``) so we stay inside aegir's
    main venv (the server hosts no model since 2026-09-08 — inference is forwarded to the federation). We do
    NOT scrub LD_LIBRARY_PATH here: the gRPC server runs in the main env and needs the cuda-driver-libs
    unmask the caller already exported.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")
    fh.write(f"\n===== engine (re)start {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
    fh.flush()
    return subprocess.Popen(
        [sys.executable, "-m", "aegir.engine.server"],
        cwd=str(_REPO), env=dict(os.environ),
        stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
    )


@dataclass
class _Attempt:
    n: int
    pid: int
    started_at: float


class EngineSupervisor:
    """Supervise the engine server child: (re)start with backoff, cap retries, log structured events.

    Injectable seams (all default to real implementations) keep this unit-testable with no subprocess/GPU:

    * ``spawn(log_path) -> proc`` — start the child; ``proc`` needs ``.pid``, ``.poll()``, ``.returncode``.
    * ``readiness_check() -> (ok: bool, detail: str)`` — called after each start; ``None`` skips the wait.
    * ``claim(gpu_ids) -> context manager`` — GPU-exclusivity guard; ``None`` skips it.
    * ``sleep(seconds)`` / ``now() -> float`` — time control for deterministic tests.
    * ``log(event: dict)`` — structured sink; defaults to JSONL to ``events_path`` + a human line on stderr.
    """

    def __init__(self, cfg: "SupervisorConfig | None" = None, *, spawn=None, readiness_check=None,
                 claim=None, gpu_ids: "list[int] | None" = None, sleep=None, now=None, log=None,
                 events_path: "Path | None" = None, server_log_path: "Path | None" = None) -> None:
        self.cfg = cfg or SupervisorConfig()
        self._server_log = server_log_path or (_LOG_DIR / "engine_server.log")
        self._events_path = events_path or (_LOG_DIR / "supervisor_events.jsonl")
        self._spawn = spawn or (lambda: _default_spawn(self._server_log))
        self._readiness_check = readiness_check  # None => skip (set by default in run() unless disabled)
        self._claim = claim
        self._gpu_ids = gpu_ids
        self._sleep = sleep or time.sleep
        self._now = now or time.time
        self._log_sink = log or self._default_log
        self._proc = None  # current child
        self._stopping = False
        self._restart_times: "deque[float]" = deque()

    # ---- structured logging -------------------------------------------------
    def _default_log(self, event: dict) -> None:
        event = {"ts": self._now(), **event}
        line = json.dumps(event)
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._events_path, "a") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001 — logging must never crash the supervisor
            pass
        # A terse human line too (so a tailing operator sees it without parsing JSON).
        ev = event.get("event", "?")
        print(f"[supervisor] {ev}: "
              + " ".join(f"{k}={v}" for k, v in event.items() if k not in ("event", "ts")),
              file=sys.stderr, flush=True)

    def log(self, event: str, **fields) -> None:
        self._log_sink({"event": event, **fields})

    # ---- backoff ------------------------------------------------------------
    def _backoff_for(self, crash_count: int) -> float:
        """Exponential backoff: base * 2**(n-1), capped. ``crash_count`` is 1 for the first crash."""
        delay = self.cfg.backoff_base_s * (2 ** max(0, crash_count - 1))
        return min(delay, self.cfg.backoff_cap_s)

    def _record_restart(self, t: float) -> int:
        """Push a restart timestamp, evict those outside the rolling window, return the in-window count."""
        self._restart_times.append(t)
        cutoff = t - self.cfg.window_s
        while self._restart_times and self._restart_times[0] < cutoff:
            self._restart_times.popleft()
        return len(self._restart_times)

    # ---- exit classification ------------------------------------------------
    @staticmethod
    def _describe_exit(rc: "int | None") -> "tuple[bool, str]":
        """Classify a child return code. Returns ``(is_clean, reason)``.

        POSIX: a process killed by signal N reports ``returncode == -N``. rc 0 is a clean exit. Anything
        else (positive rc, or a signal we didn't send) is a crash worth restarting.
        """
        if rc is None:
            return False, "still running"
        if rc == 0:
            return True, "clean exit (rc=0)"
        if rc < 0:
            sig = -rc
            name = signal.Signals(sig).name if sig in {s.value for s in signal.Signals} else f"SIG{sig}"
            return False, f"killed by {name} (rc={rc})"
        return False, f"non-zero exit (rc={rc})"

    def _tail_server_log(self, n: int = 20) -> str:
        try:
            return "\n".join(self._server_log.read_text(errors="ignore").splitlines()[-n:])
        except Exception:  # noqa: BLE001
            return ""

    # ---- lifecycle ----------------------------------------------------------
    def _start_once(self, attempt_n: int) -> _Attempt:
        proc = self._spawn()
        att = _Attempt(n=attempt_n, pid=getattr(proc, "pid", -1), started_at=self._now())
        self._proc = proc
        self.log("start", attempt=attempt_n, pid=att.pid, server_log=str(self._server_log))
        if self._readiness_check is not None:
            ok, detail = self._readiness_check()
            self.log("readiness", attempt=attempt_n, ok=ok, detail=detail)
        return att

    def _wait_for_exit(self) -> "int | None":
        """Poll the child until it exits OR a stop was requested. Returns the return code (or None if we
        were asked to stop while it was still running — caller then performs the clean killpg shutdown)."""
        while not self._stopping:
            rc = self._proc.poll()
            if rc is not None:
                return rc
            self._sleep(self.cfg.poll_interval_s)
        return None

    def shutdown_child(self, sig: int = signal.SIGTERM) -> None:
        """Clean shutdown of the current child: killpg the whole process group (so vLLM TP workers die),
        mirroring ``server._stop`` — never orphan child processes."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        pid = proc.pid
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with _suppress():
                proc.send_signal(sig)
        # Give the group a moment to drain (the server's _stop also kills its vLLM children), then SIGKILL.
        deadline = self._now() + 15.0
        while self._now() < deadline:
            if proc.poll() is not None:
                self.log("child_stopped", pid=pid, rc=proc.returncode)
                return
            self._sleep(0.5)
        with _suppress():
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        self.log("child_killed", pid=pid, note="SIGKILL after grace")

    def request_stop(self, *_sig) -> None:
        """Signal handler: ask the supervise loop to stop and tear the child down cleanly."""
        if self._stopping:
            return
        self._stopping = True
        self.log("stop_requested")
        self.shutdown_child()

    def run(self) -> int:
        """The supervise loop. Returns a process exit code (0 = clean stop, non-zero = gave up on a crash
        loop). Installs SIGINT/SIGTERM handlers when run on the main thread (skipped under test injection)."""
        # Default the readiness check to the real probe unless the caller injected/disabled it.
        if self._readiness_check is None and getattr(self, "_use_default_readiness", True):
            self._readiness_check = self._real_readiness

        # GPU guard wraps the WHOLE supervised lifetime: claim once, hold across restarts.
        guard = self._enter_guard()
        try:
            with _maybe_install_signals(self.request_stop):
                return self._supervise()
        finally:
            with _suppress():
                guard.__exit__(None, None, None)

    def _enter_guard(self):
        """Enter the GPU-exclusivity guard (or a no-op if disabled / no claim injected)."""
        if self._claim is not None:
            gpu_ids = self._gpu_ids if self._gpu_ids is not None else []
            cm = self._claim(gpu_ids)
            cm.__enter__()
            self.log("gpu_claimed", gpus=gpu_ids)
            return cm
        if self.cfg.gpu_guard and getattr(self, "_use_default_guard", True):
            from aegir.engine.gpu_guard import GpuClaimError, claim_gpus, engine_gpu_ids
            gpu_ids = self._gpu_ids if self._gpu_ids is not None else engine_gpu_ids()
            if not gpu_ids:
                self.log("gpu_claim_skipped", reason="aegir hosts no model; inference is forwarded")
                return _NoopCtx()
            try:
                cm = claim_gpus(gpu_ids)
                cm.__enter__()
            except GpuClaimError as e:
                self.log("gpu_claim_failed", gpus=gpu_ids, error=str(e))
                raise
            self.log("gpu_claimed", gpus=gpu_ids)
            return cm
        return _NoopCtx()

    def _real_readiness(self) -> "tuple[bool, str]":
        from aegir.engine.readiness import Readiness, wait_for_ready
        rep = wait_for_ready(self.cfg.capability, level=Readiness.SERVING,
                             timeout=self.cfg.readiness_timeout_s)
        return rep.ok, rep.detail

    def _supervise(self) -> int:
        attempt = 0
        crash_streak = 0
        while not self._stopping:
            attempt += 1
            self._start_once(attempt)
            rc = self._wait_for_exit()

            if self._stopping:
                # A stop was requested mid-run; child already torn down by request_stop/shutdown_child.
                self.shutdown_child()
                self.log("supervisor_stopped", reason="signal")
                return 0

            is_clean, reason = self._describe_exit(rc)
            if is_clean:
                self.log("exit_clean", attempt=attempt, rc=rc, reason=reason)
                return 0

            crash_streak += 1
            in_window = self._record_restart(self._now())
            tail = self._tail_server_log()
            self.log("exit_crash", attempt=attempt, rc=rc, reason=reason,
                     crash_streak=crash_streak, restarts_in_window=in_window,
                     window_s=self.cfg.window_s, log_tail=tail[-1500:] if tail else "")

            if in_window > self.cfg.max_retries:
                self.log("giveup", reason=f"{in_window} restarts within {self.cfg.window_s:.0f}s "
                         f"exceeds max_retries={self.cfg.max_retries}", attempt=attempt)
                return 1

            delay = self._backoff_for(crash_streak)
            self.log("backoff", seconds=delay, next_attempt=attempt + 1, crash_streak=crash_streak)
            # Sleep in poll-sized slices so a stop signal during backoff is honored promptly.
            slept = 0.0
            while slept < delay and not self._stopping:
                step = min(self.cfg.poll_interval_s, delay - slept)
                self._sleep(step)
                slept += step
        self.log("supervisor_stopped", reason="signal")
        return 0


# ---- small context helpers (avoid importing contextlib for two uses) --------
class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


class _NoopCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _maybe_install_signals:
    """Install SIGINT/SIGTERM -> ``handler`` on the main thread; no-op (and restore) otherwise.

    Signal handlers can only be set from the main thread, and tests drive ``run()`` off-thread or want their
    own control — so this degrades to a no-op when that raises, and always restores the prior handlers.
    """

    def __init__(self, handler) -> None:
        self._handler = handler
        self._prev: dict = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):  # not main thread / unsupported
                pass
        return self

    def __exit__(self, *exc):
        for sig, prev in self._prev.items():
            with _suppress():
                signal.signal(sig, prev)
        return False


def _build_config_from_args(argv: "list[str] | None" = None) -> "tuple[SupervisorConfig, bool]":
    p = argparse.ArgumentParser(description="Supervise the Aegir engine server with restart + backoff.")
    p.add_argument("--max-retries", type=int, default=None,
                   help="max crash-restarts within the rolling window before giving up")
    p.add_argument("--backoff-base", type=float, default=None, help="initial backoff seconds (doubles)")
    p.add_argument("--backoff-cap", type=float, default=None, help="max backoff seconds")
    p.add_argument("--window", type=float, default=None, help="rolling retry-count window (seconds)")
    p.add_argument("--readiness-timeout", type=float, default=None,
                   help="seconds to wait for SERVING after each (re)start")
    p.add_argument("--no-gpu-guard", action="store_true", help="skip the GPU-exclusivity guard")
    p.add_argument("--no-readiness", action="store_true", help="skip the readiness wait after each start")
    a = p.parse_args(argv)
    cfg = SupervisorConfig()
    if a.max_retries is not None:
        cfg.max_retries = a.max_retries
    if a.backoff_base is not None:
        cfg.backoff_base_s = a.backoff_base
    if a.backoff_cap is not None:
        cfg.backoff_cap_s = a.backoff_cap
    if a.window is not None:
        cfg.window_s = a.window
    if a.readiness_timeout is not None:
        cfg.readiness_timeout_s = a.readiness_timeout
    if a.no_gpu_guard:
        cfg.gpu_guard = False
    return cfg, a.no_readiness


def main(argv: "list[str] | None" = None) -> int:
    cfg, no_readiness = _build_config_from_args(argv)
    sup = EngineSupervisor(cfg)
    if no_readiness:
        sup._use_default_readiness = False  # type: ignore[attr-defined]
        sup._readiness_check = None
    return sup.run()


if __name__ == "__main__":
    raise SystemExit(main())
