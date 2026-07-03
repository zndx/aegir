"""GPU-exclusivity guard — stop two engine instances from claiming the same GPUs.

The engine pins vLLM to a contiguous GPU range starting at :data:`FIRST_GPU` (``AEGIR_ENGINE_GPU0``),
``tensor_parallel_size`` wide (see ``vllm_manager._launch``). Launching a *second* engine against the
same range silently double-commits VRAM: the new vLLM OOMs partway through model load (or, worse, both
limp along thrashing the KV cache), and a multi-day derivation run dies hours in. This guard makes that
failure FAST and LOUD at startup instead.

Two independent checks (belt **and** braces), combined by :func:`claim_gpus`:

* **lock file** — a ``filelock`` per GPU set under ``AEGIR_ENGINE_LOCK_DIR`` (default ``/tmp/aegir-engine``).
  Cooperative: only processes that go through this guard respect it, but it is the cheap first line and it
  records *who* holds the GPUs (pid + worktree role + timestamp) for a human-readable conflict message.
  ``filelock`` is advisory and auto-released if the holder dies (the OS drops the ``flock``), so a crashed
  engine does not wedge the lock.
* **nvidia-smi compute-process probe** — authoritative cross-process truth: queries
  ``--query-compute-apps=pid,used_memory,gpu_uuid`` and maps it back to GPU *indices* via
  ``--query-gpu=index,uuid``. If a foreign process already holds ≥ ``min_mib`` on a target GPU we refuse,
  regardless of whether it took the lock (covers a hand-launched vLLM, a stray training job, another user).

Ties into the worktree-role convention (``AEGIR_WORKTREE_ROLE`` / ``bin/detect-worktree-role.sh``): the
role is recorded in the lock payload and surfaced in conflict messages, so "primary already owns GPUs 0-3"
reads naturally in a multi-worktree layout. This guard does NOT itself gate on role — owning the engine is
a deployment choice — it just reports it.

Usage (supervisor / server startup)::

    from aegir.engine.gpu_guard import claim_gpus
    with claim_gpus([0, 1, 2, 3]) as lock:   # raises GpuClaimError if already claimed
        serve(...)                            # lock auto-released on exit

Test seam: ``_nvidia_compute_apps`` / ``_nvidia_gpu_uuids`` are module-level and monkeypatchable, so the
probe logic is unit-tested without a GPU. ``FileLock`` is real (cheap, tmpdir) in tests.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

# Default min VRAM (MiB) a foreign process must hold on a GPU for us to consider it "in use". A few MiB of
# residual CUDA context from a dead process should not block; an active vLLM holds many GiB. 512 MiB splits
# those cleanly. Override with AEGIR_ENGINE_GPU_MIN_MIB.
_DEFAULT_MIN_MIB = int(os.environ.get("AEGIR_ENGINE_GPU_MIN_MIB", "512"))
# SHARED cross-project lease dir (federation prep, adopted from Atelier's proposal 2026-07-03): all
# zndx engines (aegir :50151, atelier :50251, gaius :50051) drop their per-GPU-set advisory locks in
# ONE dir, so the cooperative layer is mutual across projects — previously each engine's locks lived
# in a per-project dir and only the nvidia-smi probe protected co-tenancy. Lock filenames are
# engine-agnostic (gpu-<set>.lock); the owner payload names the project. Effective at next restart.
_LOCK_DIR = Path(os.environ.get("AEGIR_ENGINE_LOCK_DIR", "/tmp/zndx-gpu-leases"))
_WORKTREE_ROLE = os.environ.get("AEGIR_WORKTREE_ROLE", "primary")
_PROJECT = "aegir"


class GpuClaimError(RuntimeError):
    """Target GPUs are already claimed (by a lock holder and/or a foreign compute process)."""


@dataclass(frozen=True)
class GpuUser:
    """A foreign compute process found on a target GPU by the nvidia-smi probe."""

    gpu_index: int
    pid: int
    used_mib: int

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"GPU{self.gpu_index} held by pid {self.pid} ({self.used_mib} MiB)"


def _run(cmd: list[str], timeout: float = 10.0) -> "str | None":
    """Run ``cmd``, return stdout text, or ``None`` if the binary is missing / it fails / times out.

    A missing or failing nvidia-smi must DEGRADE the probe (skip it), never crash the engine — the lock
    file remains as the cooperative fallback.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _nvidia_compute_apps() -> "list[tuple[int, int, str]] | None":
    """``[(pid, used_mib, gpu_uuid), ...]`` for every compute process, or ``None`` if nvidia-smi is missing.

    ``--query-compute-apps`` reports a ``gpu_uuid`` (NOT an index), so :func:`_foreign_users` resolves each
    uuid to its GPU index via the parallel :func:`_nvidia_gpu_uuids` call before matching against targets.
    """
    txt = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits"])
    if txt is None:
        return None
    rows: list[tuple[int, int, str]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1])), parts[2]))
        except ValueError:
            continue  # "[N/A]" or "[Not Supported]" — skip the row, don't fail the probe
    return rows


def _nvidia_gpu_uuids() -> "dict[str, int] | None":
    """``{gpu_uuid: index}`` map, or ``None`` if nvidia-smi is unavailable."""
    txt = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    if txt is None:
        return None
    out: dict[str, int] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            out[parts[1]] = int(parts[0])
        except ValueError:
            continue
    return out


def _foreign_users(target: list[int], min_mib: int,
                   self_pids: "set[int] | None" = None) -> "list[GpuUser] | None":
    """Compute processes on ``target`` GPU indices holding ≥ ``min_mib``, excluding ``self_pids``.

    Returns ``None`` (probe unavailable → degrade to lock-only) if either nvidia-smi query fails.
    """
    apps = _nvidia_compute_apps()
    uuids = _nvidia_gpu_uuids()
    if apps is None or uuids is None:
        return None
    self_pids = self_pids or set()
    want = set(target)
    found: list[GpuUser] = []
    for pid, used_mib, uuid in apps:
        idx = uuids.get(uuid)
        if idx is None or idx not in want:
            continue
        if pid in self_pids or used_mib < min_mib:
            continue
        found.append(GpuUser(gpu_index=idx, pid=pid, used_mib=used_mib))
    return found


def _lock_path(target: list[int]) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(g) for g in sorted(target))
    return _LOCK_DIR / f"gpu-{tag}.lock"


def probe_gpus(target: list[int], *, min_mib: int = _DEFAULT_MIN_MIB,
               self_pids: "set[int] | None" = None) -> "list[GpuUser]":
    """Non-locking read of who (foreign) currently holds ``target`` GPUs. Empty if free / probe degraded.

    Used by the readiness/status surface and as the precheck inside :func:`claim_gpus`.
    """
    users = _foreign_users(target, min_mib, self_pids=self_pids)
    return users or []


@contextlib.contextmanager
def claim_gpus(target: list[int], *, min_mib: int = _DEFAULT_MIN_MIB,
               lock_timeout: float = 0.0, self_pids: "set[int] | None" = None,
               role: "str | None" = None):
    """Claim exclusive use of ``target`` GPU indices for the duration of the ``with`` block.

    Acquires the per-GPU-set lock file AND verifies (via nvidia-smi) that no foreign process already holds
    the GPUs. Raises :class:`GpuClaimError` immediately (``lock_timeout=0``) if either check fails, with a
    message naming the conflicting holder. On a machine without nvidia-smi the compute probe is skipped and
    only the cooperative lock applies.

    ``self_pids`` excludes our own already-running vLLM workers from the foreign check (used on restart,
    where the supervisor knows its prior children). ``role`` defaults to ``AEGIR_WORKTREE_ROLE``.
    """
    if not target:
        raise ValueError("claim_gpus: empty GPU list")
    role = role or _WORKTREE_ROLE
    lock_path = _lock_path(target)
    info_path = lock_path.with_suffix(".owner.json")
    lock = FileLock(str(lock_path), timeout=lock_timeout)
    try:
        lock.acquire(timeout=lock_timeout)
    except Timeout:
        holder = _read_owner(info_path)
        raise GpuClaimError(
            f"GPUs {target} already locked by {holder} (lock {lock_path}). "
            f"Another engine instance is running, or a previous one did not release the lock."
        ) from None

    # Lock held — now the authoritative cross-process check.
    foreign = probe_gpus(target, min_mib=min_mib, self_pids=self_pids)
    if foreign:
        lock.release()
        detail = "; ".join(str(u) for u in foreign)
        raise GpuClaimError(
            f"GPUs {target} already in use by a foreign compute process: {detail}. "
            f"Refusing to double-commit VRAM. Stop the other process or set AEGIR_ENGINE_GPU0 to a free range."
        )

    _write_owner(info_path, target, role)
    try:
        yield lock
    finally:
        with contextlib.suppress(Exception):
            info_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            lock.release()


def _write_owner(path: Path, target: list[int], role: str) -> None:
    payload = {"pid": os.getpid(), "project": _PROJECT, "role": role, "gpus": target,
               "ts": time.time(), "host": os.uname().nodename}
    with contextlib.suppress(Exception):
        path.write_text(json.dumps(payload))


def _read_owner(path: Path) -> str:
    try:
        d = json.loads(path.read_text())
        age = max(0, int(time.time() - d.get("ts", 0)))
        return (f"pid {d.get('pid')} (project={d.get('project', '?')}, role={d.get('role')}, "
                f"host={d.get('host')}, gpus={d.get('gpus')}, {age}s ago)")
    except Exception:  # noqa: BLE001
        return "an unknown process"


def engine_gpu_ids(first_gpu: "int | None" = None, tp: "int | None" = None) -> list[int]:
    """The contiguous GPU index range the engine's ``instruct`` capability will pin (mirror of the
    ``vllm_manager._launch`` arithmetic), for the supervisor/server to claim up-front.
    """
    from aegir.engine.config import CAPABILITY_MODELS, FIRST_GPU
    first = FIRST_GPU if first_gpu is None else first_gpu
    width = tp if tp is not None else CAPABILITY_MODELS["instruct"].tensor_parallel_size
    return list(range(first, first + width))
