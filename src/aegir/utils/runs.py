"""Training-run artifact writer: metadata + metrics (two JSON sidecars).

Each training run lands in ``{runs_root}/{run_id}/`` with:

  metadata.json  — immutable: task, model_size, arch_layout, d_model,
                   seed, num_params, git SHA (short+long), utc_start,
                   utc_end, argv, host, env-summary.
  metrics.json   — per-epoch list + ``final`` summary (best val macro_f1):
                   {epochs: [{epoch, train_loss, val_loss, micro_f1,
                   macro_f1, boundary: {stage0_mean_F, ...}}, ...],
                   final: {...}}

Plots are **not** pre-rendered. The HoloViews builders below (``_loss_curve`` /
``_f1_curve`` / ``_boundary_curve``) are rendered LIVE off metrics.json by
``aegir.viz.runs_app`` over the bokeh server and embedded into the React leaderboard
(``<PanelView>``); ``available_plots`` derives which exist. See
docs/scratch/2026-06-18/195536_live_viz_spine.md.

Why JSON sidecars, not a database: the leaderboard is read-only, one row per run;
git-diffable, removable without migration. Postgres is provisioned but unused by the
leaderboard until M2 (JOINs / user state).
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── run_id construction ────────────────────────────────────────


def _short_sha(cwd: Path | None = None) -> str:
    """Short git SHA of HEAD, or ``nogit`` if unavailable.

    Safe to call in sandboxed / checked-out-only scenarios: failure returns a
    deterministic sentinel rather than raising.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True, text=True, cwd=str(cwd) if cwd else None,
            timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def _full_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(cwd) if cwd else None,
            timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def _git_dirty(cwd: Path | None = None) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(cwd) if cwd else None,
            timeout=2, check=False,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except (OSError, subprocess.SubprocessError):
        return False


def _utc_iso(compact: bool = True) -> str:
    """UTC timestamp. Compact form strips separators for safe filename use."""
    if compact:
        return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(task: str, model_size: str, short_sha: str | None = None) -> str:
    """Deterministic, sortable ``run_id`` string.

    Format: ``{UTC_ISO_COMPACT}_{short_sha}_{model_size}_{task}``.
    Lexical sort is chronological. Collisions require same-second concurrency
    *and* same task/model — rare; training launches don't overlap on one GPU.
    """
    if short_sha is None:
        short_sha = _short_sha()
    safe_task = task.replace("/", "_")
    safe_size = model_size.replace("/", "_")
    return f"{_utc_iso()}_{short_sha}_{safe_size}_{safe_task}"


# ── HoloViews plot builders (rendered live by aegir.viz.runs_app over the bokeh server) ──────────


def _loss_curve(epochs: list[dict]) -> Any:
    """Train + val loss across epochs as overlaid curves."""
    import holoviews as hv
    xs = [e["epoch"] for e in epochs]
    train = [e.get("train_loss", float("nan")) for e in epochs]
    val = [e.get("val_loss", float("nan")) for e in epochs]
    curve_train = hv.Curve(list(zip(xs, train)), "epoch", "loss", label="train")
    curve_val = hv.Curve(list(zip(xs, val)), "epoch", "loss", label="val")
    return (curve_train * curve_val).opts(
        title="Loss", width=640, height=320, legend_position="top_right",
    )


def _f1_curve(epochs: list[dict]) -> Any:
    import holoviews as hv
    xs = [e["epoch"] for e in epochs]
    micro = [e.get("micro_f1", float("nan")) for e in epochs]
    macro = [e.get("macro_f1", float("nan")) for e in epochs]
    curve_micro = hv.Curve(list(zip(xs, micro)), "epoch", "F1", label="micro")
    curve_macro = hv.Curve(list(zip(xs, macro)), "epoch", "F1", label="macro")
    return (curve_micro * curve_macro).opts(
        title="Validation F1", width=640, height=320, legend_position="bottom_right",
    )


def _boundary_curve(epochs: list[dict], stage: int) -> Any | None:
    """mean_F and mean_G for a single routing stage; None if stage absent."""
    import holoviews as hv
    xs: list[int] = []
    f_vals: list[float] = []
    g_vals: list[float] = []
    for e in epochs:
        b = e.get("boundary") or {}
        f_key = f"stage{stage}_mean_F"
        g_key = f"stage{stage}_mean_G"
        if f_key not in b:
            continue
        xs.append(e["epoch"])
        f_vals.append(b[f_key])
        g_vals.append(b[g_key])
    if not xs:
        return None
    curve_f = hv.Curve(list(zip(xs, f_vals)), "epoch", "rate", label="mean_F (selected)")
    curve_g = hv.Curve(list(zip(xs, g_vals)), "epoch", "rate", label="mean_G (prob)")
    return (curve_f * curve_g).opts(
        title=f"Chunking — stage {stage}",
        width=640, height=320, legend_position="top_right",
    )


# ── RunArtifacts ──────────────────────────────────────────────


@dataclass
class RunArtifacts:
    """Writer for one training run's on-disk artifacts.

    Typical use inside ``train.py``::

        run = RunArtifacts.start(args, runs_root=Path("outputs/runs"))
        run.set_num_params(num_params)
        for epoch in ...:
            ...
            run.add_epoch_metrics({...})
        run.finalize(final_summary={...})

    ``start`` captures ``utc_start``, argv, cwd, host, git SHA. ``finalize``
    re-captures end time and renders all plots.
    """

    run_id: str
    run_dir: Path
    metadata: dict
    epoch_metrics: list[dict] = field(default_factory=list)
    _finalized: bool = False

    @classmethod
    def start(
        cls,
        args,
        runs_root: Path,
        *,
        project_root: Path | None = None,
        task: str | None = None,
        model_size: str | None = None,
    ) -> RunArtifacts:
        """Build + register a new run.

        Args is expected to be the argparse Namespace from train.py (for argv
        capture); ``task`` / ``model_size`` fall back to ``args.task`` /
        ``args.model_size`` when not explicitly passed.
        """
        task_str: str = str(task or getattr(args, "task", None) or "unknown")
        size_str: str = str(model_size or getattr(args, "model_size", None) or "unknown")
        project_root = project_root or Path.cwd()
        short = _short_sha(project_root)
        run_id = make_run_id(task=task_str, model_size=size_str, short_sha=short)
        run_dir = Path(runs_root) / run_id
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)

        metadata = {
            "run_id": run_id,
            "task": task_str,
            "model_size": size_str,
            "seed": getattr(args, "seed", None),
            "vocab_size": getattr(args, "vocab_size", None),
            "batch_size": getattr(args, "batch_size", None),
            "lr": getattr(args, "lr", None),
            "epochs_requested": getattr(args, "epochs", None),
            "max_length": getattr(args, "max_length", None),
            "max_context_cols": getattr(args, "max_context_cols", None),
            "downsample_factor": getattr(args, "downsample_factor", None),
            "lambda_lb": getattr(args, "lambda_lb", None),
            "amp": getattr(args, "amp", None),
            "smoke_test": getattr(args, "smoke_test", False),
            "git_short_sha": short,
            "git_full_sha": _full_sha(project_root),
            "git_dirty": _git_dirty(project_root),
            "utc_start": _utc_iso(compact=False),
            "argv": sys.argv,
            "cwd": str(project_root),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "num_params": None,
            "utc_end": None,
        }
        cls._atomic_write_json(run_dir / "metadata.json", metadata)
        return cls(run_id=run_id, run_dir=run_dir, metadata=metadata)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
        os.replace(tmp, path)

    def set_num_params(self, n: int) -> None:
        self.metadata["num_params"] = int(n)
        self._atomic_write_json(self.run_dir / "metadata.json", self.metadata)

    def add_epoch_metrics(self, row: dict) -> None:
        """Append one epoch's metrics. Tolerant of partial dicts."""
        # Promote boundary to a nested dict so callers can either pass it flat
        # or pre-nested — training loop currently returns flat keys plus a
        # ``boundary`` dict; we accept either shape.
        boundary = row.get("boundary") if isinstance(row.get("boundary"), dict) else None
        normalized = {
            "epoch": int(row.get("epoch", len(self.epoch_metrics) + 1)),
            "train_loss": _as_float(row.get("train_loss") or row.get("loss")),
            "train_task_loss": _as_float(row.get("train_task_loss") or row.get("task_loss")),
            "train_lb_loss": _as_float(row.get("train_lb_loss") or row.get("lb_loss")),
            "val_loss": _as_float(row.get("val_loss")),
            "micro_f1": _as_float(row.get("micro_f1")),
            "macro_f1": _as_float(row.get("macro_f1")),
            "boundary": {k: _as_float(v) for k, v in (boundary or {}).items()},
            "wall_seconds": _as_float(row.get("wall_seconds")),
        }
        self.epoch_metrics.append(normalized)
        # Incrementally persist so a killed run still has partial metrics.
        self._atomic_write_json(
            self.run_dir / "metrics.json",
            {"epochs": self.epoch_metrics, "final": None},
        )

    def finalize(self, final_summary: dict | None = None) -> None:
        """Write end timestamp, final summary, and render static Bokeh plots."""
        if self._finalized:
            return
        self.metadata["utc_end"] = _utc_iso(compact=False)
        self._atomic_write_json(self.run_dir / "metadata.json", self.metadata)

        final_payload = {
            "epochs": self.epoch_metrics,
            "final": final_summary or self._auto_final_summary(),
        }
        self._atomic_write_json(self.run_dir / "metrics.json", final_payload)
        self._finalized = True

    def _auto_final_summary(self) -> dict:
        if not self.epoch_metrics:
            return {}

        def _key(r: dict) -> float:
            v = r.get("macro_f1")
            return float(v) if v is not None else -1.0

        best = max(self.epoch_metrics, key=_key)
        last = self.epoch_metrics[-1]
        return {
            "best_epoch": best["epoch"],
            "best_val_macro_f1": best.get("macro_f1"),
            "best_val_micro_f1": best.get("micro_f1"),
            "last_epoch": last["epoch"],
            "last_train_loss": last.get("train_loss"),
            "last_val_loss": last.get("val_loss"),
            "num_epochs_completed": len(self.epoch_metrics),
        }



def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── read-side helpers (used by gateway) ───────────────────────


def iter_runs(runs_root: Path) -> list[Path]:
    """List run directories (newest first by lexical order of ``run_id``)."""
    if not runs_root.exists():
        return []
    return sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and (p / "metadata.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


def load_run_summary(run_dir: Path) -> dict:
    """Combine metadata.json + metrics.json.final for one run into a flat row."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"epochs": [], "final": None}
    final = metrics.get("final") or {}
    return {
        "run_id": meta.get("run_id"),
        "task": meta.get("task"),
        "model_size": meta.get("model_size"),
        "num_params": meta.get("num_params"),
        "num_epochs": final.get("num_epochs_completed") or len(metrics.get("epochs", [])),
        "best_val_macro_f1": final.get("best_val_macro_f1"),
        "best_val_micro_f1": final.get("best_val_micro_f1"),
        "last_train_loss": final.get("last_train_loss"),
        "last_val_loss": final.get("last_val_loss"),
        "git_short_sha": meta.get("git_short_sha"),
        "utc_start": meta.get("utc_start"),
        "utc_end": meta.get("utc_end"),
        "smoke_test": meta.get("smoke_test"),
    }


def available_plots(epochs: list[dict]) -> list[str]:
    """Plot names renderable live by ``aegir.viz.runs_app`` from a run's metrics — the leaderboard
    gates its ``<PanelView>``s on this. Derived from metrics (loss/f1 + each routing stage with
    boundary data), not from pre-rendered files (those are gone — plots render live now)."""
    if not epochs:
        return []
    stages: set[int] = set()
    for e in epochs:
        for k in (e.get("boundary") or {}):
            if k.startswith("stage") and k.endswith("_mean_F"):
                try:
                    stages.add(int(k[len("stage"):k.index("_mean_F")]))
                except ValueError:
                    continue
    return ["loss", "f1"] + [f"boundary_stage{s}" for s in sorted(stages)]


def load_run_detail(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "metadata.json").read_text())
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"epochs": [], "final": None}
    return {"metadata": meta, "metrics": metrics, "plots": available_plots(metrics.get("epochs", []))}
