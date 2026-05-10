"""Checkpoint metadata + sidecar callback for the P5 GRPO run.

The HuggingFace ``Trainer`` (inherited by ``trl.GRPOTrainer``)
covers the model + LoRA adapters + AdamW state + LR scheduler +
RNG + AMP scaler. This module covers the *aegir-specific* state
that HF doesn't know about:

- **Catalog version pin** — which catalog the policy was trained
  against. Resuming with a newer catalog silently changes the
  reward landscape; we catch that at restart.
- **Locked verifier weight hash** — records ``{a=0.50, b=0.05,
  c=0.45}`` identity so a config drift can't silently retune
  the reward signal.
- **T_I + null-stats path** — both contribute to R_D; pin them
  so we know what corpus the policy aligned to.
- **Run id** — unique per launch; used for trace correlation
  with downstream artifacts (held-out eval, SAE log).

A ``SidecarCallback`` plugs into the HF ``Trainer`` callback
list and writes ``metadata.json`` + flushes the SAE log on
every checkpoint save event. ``verify_resume_metadata(...)``
runs at resume time and refuses to continue if the catalog
version or locked-weight hash drifted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aegir.ontology.verifier import W_R_B, W_R_C, W_R_D

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Settings that map onto HF ``TrainingArguments`` fields plus
    aegir-specific sidecar paths.

    Defaults target a 200 GPU-hour run with checkpoints every 50
    GRPO iterations (~30-60 min wall-clock per checkpoint at
    expected real-policy throughput) and a rolling retention of
    3 most-recent checkpoints. LoRA adapters are small enough
    that 3 is comfortable.

    **Disk routing**. ``output_dir`` defaults to
    ``/raid/checkpoints/p5`` because the system drive is space-
    constrained (~15 GB free) and cannot hold checkpoints,
    optimizer state, or training logs for the duration of the
    200 GPU-hour run. ``/raid`` has the HF cache symlink at
    ``~/.cache/huggingface → /raid/cache/huggingface`` so model
    weight downloads land on /raid as well.
    """
    output_dir: str = "/raid/checkpoints/p5"
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    # In-loop eval is disabled by default. The held-out 50 set
    # is consumed *post-training* via ``aegir.rl.eval.evaluate``;
    # wiring it as a TRL ``eval_dataset`` would require either a
    # GRPO-compatible eval (which TRL doesn't have a clean recipe
    # for) or a separate forward-pass loss eval (which doesn't
    # capture our composition-quality signal). Override to
    # ``"steps"`` only with an ``eval_dataset`` plumbed through.
    eval_strategy: str = "no"
    eval_steps: int = 50
    logging_strategy: str = "steps"
    logging_steps: int = 5
    resume_from_checkpoint: str | bool | None = None
    metadata_filename: str = "aegir_metadata.json"
    sae_log_filename: str = "sae_features.jsonl"
    grpo_metrics_filename: str = "grpo_metrics.jsonl"


@dataclass
class RunMetadata:
    """Aegir-side checkpoint metadata. Persisted alongside every
    HF checkpoint; verified at resume time."""
    run_id: str
    catalog_path: str
    catalog_version: str
    catalog_n_templates: int
    catalog_n_verbalized: int
    catalog_n_complex: int
    locked_weights: dict[str, float]
    locked_weights_hash: str
    t_i_cache_path: str
    null_stats_path: str
    null_stats_hash: str
    started_at: float = field(default_factory=time.time)
    last_checkpoint_at: float = field(default_factory=time.time)
    last_checkpoint_step: int = 0
    p5_brief_commit: str | None = None


def hash_locked_weights(w_b: float = W_R_B, w_c: float = W_R_C, w_d: float = W_R_D) -> str:
    """Stable hash over the locked aggregation weights. Any drift
    from the C1-locked values flips this hash."""
    payload = json.dumps({"a": w_b, "b": w_c, "c": w_d}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    """SHA-256 prefix over a file's contents. Used to pin
    ``T_I.pkl`` and ``null_stats.json`` so a quiet refit between
    pause and resume is detected."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_run_metadata(
    catalog_path: str,
    t_i_cache_path: str,
    null_stats_path: str,
    p5_brief_commit: str | None = None,
) -> RunMetadata:
    """Construct a fresh ``RunMetadata`` from current state.
    Reads the catalog only enough to capture summary counts."""
    cat_data = json.loads(Path(catalog_path).read_text())
    templates = cat_data.get("templates", [])
    n_verbalized = sum(1 for t in templates if t.get("verbal_template"))
    n_complex = sum(1 for t in templates if t.get("is_complex"))

    return RunMetadata(
        run_id=str(uuid.uuid4()),
        catalog_path=str(catalog_path),
        catalog_version=cat_data.get("version", "unknown"),
        catalog_n_templates=len(templates),
        catalog_n_verbalized=n_verbalized,
        catalog_n_complex=n_complex,
        locked_weights={"a": W_R_B, "b": W_R_C, "c": W_R_D},
        locked_weights_hash=hash_locked_weights(),
        t_i_cache_path=str(t_i_cache_path),
        null_stats_path=str(null_stats_path),
        null_stats_hash=hash_file(null_stats_path),
        p5_brief_commit=p5_brief_commit,
    )


def write_metadata(metadata: RunMetadata, checkpoint_dir: str | Path,
                   filename: str = "aegir_metadata.json") -> Path:
    """Write metadata JSON into a checkpoint directory."""
    cp = Path(checkpoint_dir)
    cp.mkdir(parents=True, exist_ok=True)
    out = cp / filename
    out.write_text(json.dumps(asdict(metadata), indent=2))
    return out


def read_metadata(checkpoint_dir: str | Path,
                  filename: str = "aegir_metadata.json") -> RunMetadata:
    """Load metadata from a checkpoint directory."""
    p = Path(checkpoint_dir) / filename
    raw = json.loads(p.read_text())
    return RunMetadata(**raw)


def verify_resume_metadata(
    checkpoint_dir: str | Path,
    catalog_path: str,
    t_i_cache_path: str,
    null_stats_path: str,
    strict: bool = True,
    filename: str = "aegir_metadata.json",
) -> RunMetadata:
    """Sanity-check resume conditions. Raises ``ValueError`` if
    ``strict`` and any of (catalog version, locked-weight hash,
    null-stats hash) drifted. Returns the loaded metadata on
    success.

    **Default policy: strict=True.** Any drift in catalog
    contents, locked verifier weights, or null-stats is treated
    as a surprise and refused. This preserves determinism and
    reproducibility across restarts, which the v0.5 brief's
    RLVR claims rely on. Silent acceptance of a changed reward
    landscape would invalidate the run's identity.

    Override only with documented intent:

    - Pause/resume mid-run → ``strict=True`` (default). Any
      drift fails loud.
    - Deliberate catalog extension mid-run (e.g., hot-loading
      Batch 8 templates) → ``strict=False`` only when the
      operator records the rationale in the run's
      ``RunMetadata.notes`` field (or its successor) and
      acknowledges the policy's effective objective has
      changed. Such overrides should be rare and explicit.
    """
    meta = read_metadata(checkpoint_dir, filename=filename)

    cat_data = json.loads(Path(catalog_path).read_text())
    cat_version_now = cat_data.get("version", "unknown")
    cat_n_now = len(cat_data.get("templates", []))
    locked_hash_now = hash_locked_weights()
    null_hash_now = hash_file(null_stats_path)

    drift: list[str] = []
    if cat_version_now != meta.catalog_version:
        drift.append(
            f"catalog version: checkpoint={meta.catalog_version!r} now={cat_version_now!r}"
        )
    if cat_n_now != meta.catalog_n_templates:
        drift.append(
            f"catalog n_templates: checkpoint={meta.catalog_n_templates} now={cat_n_now}"
        )
    if locked_hash_now != meta.locked_weights_hash:
        drift.append(
            f"locked-weights hash: checkpoint={meta.locked_weights_hash!r} now={locked_hash_now!r}"
        )
    if null_hash_now != meta.null_stats_hash:
        drift.append(
            f"null-stats hash: checkpoint={meta.null_stats_hash!r} now={null_hash_now!r}"
        )
    if t_i_cache_path != meta.t_i_cache_path:
        drift.append(
            f"T_I cache path: checkpoint={meta.t_i_cache_path!r} now={t_i_cache_path!r}"
        )

    if drift:
        msg = "resume metadata drift:\n  " + "\n  ".join(drift)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    return meta


# ──────────────────────────────────────────────────────────────
# HF Trainer callback: writes the metadata sidecar + flushes the
# SAE feature log on every save event.
# ──────────────────────────────────────────────────────────────


def make_sidecar_callback(metadata: RunMetadata, sae_logger=None,
                          cfg: CheckpointConfig | None = None):
    """Return a ``TrainerCallback`` instance bound to this run's
    metadata. Lazy-imports ``transformers`` so the scaffolding
    smoke test does not require the package."""
    from transformers import TrainerCallback

    bound_cfg: CheckpointConfig = cfg if cfg is not None else CheckpointConfig()

    class SidecarCallback(TrainerCallback):
        def on_save(self, args, state, _control, **_kwargs):
            if not state.is_world_process_zero:
                return
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            metadata.last_checkpoint_at = time.time()
            metadata.last_checkpoint_step = state.global_step
            write_metadata(metadata, checkpoint_dir, filename=bound_cfg.metadata_filename)
            if sae_logger is not None:
                sae_log_path = checkpoint_dir / bound_cfg.sae_log_filename
                sae_logger.spill_to_disk(sae_log_path)
            logger.info(
                "[checkpointing] sidecar written for step %d → %s",
                state.global_step, checkpoint_dir,
            )

        def on_train_end(self, args, state, _control, **_kwargs):
            if not state.is_world_process_zero:
                return
            if sae_logger is not None and sae_logger.records:
                fallback = Path(args.output_dir) / bound_cfg.sae_log_filename
                sae_logger.spill_to_disk(fallback)

    return SidecarCallback()


def to_training_arguments(cfg: CheckpointConfig, **overrides):
    """Translate ``CheckpointConfig`` into HF ``TrainingArguments``.
    Lazy import."""
    from transformers import TrainingArguments

    args = dict(
        output_dir=cfg.output_dir,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        logging_strategy=cfg.logging_strategy,
        logging_steps=cfg.logging_steps,
    )
    if cfg.resume_from_checkpoint is not None:
        args["resume_from_checkpoint"] = cfg.resume_from_checkpoint
    args.update(overrides)
    return TrainingArguments(**args)


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` subdirectory
    under ``output_dir``, or ``None`` if no checkpoints exist."""
    p = Path(output_dir)
    if not p.exists():
        return None
    candidates = []
    for entry in p.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        candidates.append((step, entry))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]
