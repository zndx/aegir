"""GRPO rollout / scoring / advantage / policy-gradient loop.

This module wires the verifier reward signal into the GRPO
training loop. The verifier is hot-loaded per call: any catalog
update (Batches 6, 7, post-P5 fixes) is picked up automatically
on the next group rollout without restarting training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from aegir.ontology.schema import Catalog, load_catalog
from aegir.ontology.verifier import (
    CompositionEntry,
    VerifierResult,
    verify,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Hyper-parameters for the GRPO loop."""

    catalog_path: str = "src/aegir/ontology/catalog/combined.json"
    t_i_cache_path: str = "src/aegir/ontology/catalog/T_I_canonical.pkl"
    null_stats_path: str = "src/aegir/ontology/catalog/null_stats_canonical.json"
    group_size: int = 8
    learning_rate: float = 1e-5
    kl_coefficient: float = 0.04
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    num_iterations: int = 1000
    eval_every: int = 50
    advantage_normalization: str = "z_score"  # "z_score" | "centered"
    skip_r_d_during_warmup: int = 50  # use only R_A · (a*R_B + b*R_C) for first N iters
    # Append-only JSONL of GRPOMetrics, written every iteration.
    # Survives restart so a paused run can be reconstructed for
    # post-hoc analysis. ``None`` disables persistence.
    metrics_jsonl_path: str | None = None


@dataclass
class GRPOMetrics:
    iteration: int
    rewards_mean: float
    rewards_std: float
    rewards_min: float
    rewards_max: float
    advantage_mean: float
    advantage_std: float
    r_a_pass_rate: float
    n_invalid_compositions: int


@dataclass
class GRPOState:
    catalog: Catalog
    config: GRPOConfig
    metrics: list[GRPOMetrics] = field(default_factory=list)


def hot_reload_catalog(state: GRPOState) -> bool:
    """Re-read the catalog file from disk. Returns True if it
    actually changed (template count or version)."""
    new_cat = load_catalog(state.config.catalog_path)
    changed = (
        new_cat.version != state.catalog.version
        or len(new_cat.templates) != len(state.catalog.templates)
    )
    state.catalog = new_cat
    if changed:
        logger.info(
            "catalog hot-reloaded: %s (%d templates)",
            new_cat.version, len(new_cat.templates),
        )
    return changed


def score_composition(
    composition: list[CompositionEntry],
    state: GRPOState,
    iteration: int,
) -> VerifierResult:
    """Score a single composition. R_D may be skipped for early
    iterations per ``GRPOConfig.skip_r_d_during_warmup``."""
    skip_r_d = iteration < state.config.skip_r_d_during_warmup
    return verify(
        composition,
        state.catalog,
        t_i_cache_path=state.config.t_i_cache_path,
        null_stats_path=state.config.null_stats_path,
        skip_r_d=skip_r_d,
    )


def compute_group_advantages(
    rewards: list[float],
    normalization: str = "z_score",
) -> list[float]:
    """Group-relative advantages per the GRPO paper. ``z_score``
    is mean-zero / unit-variance; ``centered`` is mean-zero only."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if normalization == "centered":
        return [r - mean for r in rewards]
    var = sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)
    std = var ** 0.5 if var > 0 else 1.0
    return [(r - mean) / std for r in rewards]


def grpo_iteration(
    state: GRPOState,
    iteration: int,
    rollout_fn: Callable[[Catalog, int], list[list[CompositionEntry]]],
    policy_step_fn: Callable[
        [list[list[CompositionEntry]], list[float]], None
    ] | None = None,
) -> GRPOMetrics:
    """One GRPO iteration: rollout → score → advantages → policy step.

    ``rollout_fn`` is the policy-side sampling function (returns
    ``group_size`` candidate compositions). ``policy_step_fn``
    consumes (compositions, advantages) and applies the policy
    gradient step. Both are injected so this loop is testable
    without a real policy."""
    compositions = rollout_fn(state.catalog, state.config.group_size)
    rewards: list[float] = []
    valid_compositions: list[list[CompositionEntry]] = []
    n_invalid = 0
    for comp in compositions:
        if not comp:
            rewards.append(0.0)
            valid_compositions.append([])
            n_invalid += 1
            continue
        result = score_composition(comp, state, iteration)
        rewards.append(result.R)
        valid_compositions.append(comp)

    advantages = compute_group_advantages(
        rewards,
        normalization=state.config.advantage_normalization,
    )

    if policy_step_fn is not None:
        policy_step_fn(valid_compositions, advantages)

    n = max(len(rewards), 1)
    sorted_r = sorted(rewards)
    metrics = GRPOMetrics(
        iteration=iteration,
        rewards_mean=sum(rewards) / n,
        rewards_std=(
            sum((r - sum(rewards) / n) ** 2 for r in rewards) / max(n - 1, 1)
        ) ** 0.5,
        rewards_min=sorted_r[0] if sorted_r else 0.0,
        rewards_max=sorted_r[-1] if sorted_r else 0.0,
        advantage_mean=sum(advantages) / n,
        advantage_std=(
            sum((a - sum(advantages) / n) ** 2 for a in advantages) / max(n - 1, 1)
        ) ** 0.5,
        r_a_pass_rate=sum(1 for r in rewards if r > 0) / n,
        n_invalid_compositions=n_invalid,
    )
    state.metrics.append(metrics)
    if state.config.metrics_jsonl_path:
        _append_metrics_jsonl(state.config.metrics_jsonl_path, metrics)
    return metrics


def _append_metrics_jsonl(path: str, metrics: GRPOMetrics) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")


def load_metrics_history(path: str) -> list[GRPOMetrics]:
    """Restore the metrics history from JSONL on resume. Returns
    an empty list if the file does not yet exist."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[GRPOMetrics] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(GRPOMetrics(**obj))
    return out


def init_grpo_state(cfg: GRPOConfig) -> GRPOState:
    state = GRPOState(
        catalog=load_catalog(cfg.catalog_path),
        config=cfg,
    )
    if cfg.metrics_jsonl_path:
        state.metrics = load_metrics_history(cfg.metrics_jsonl_path)
        if state.metrics:
            logger.info(
                "GRPO metrics resumed: %d prior iterations from %s",
                len(state.metrics), cfg.metrics_jsonl_path,
            )
    return state
