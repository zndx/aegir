#!/usr/bin/env python
"""P5 scaffolding smoke test — validates the wiring without
actually loading the 27B base policy.

What this exercises:

- Catalog hot-reload through ``aegir.rl.grpo_loop.init_grpo_state``.
- Reward scoring via ``aegir.rl.grpo_loop.score_composition`` with
  the locked verifier.
- Group-advantage computation against a stub policy that returns
  a mix of good and bad compositions.
- Constrained-decoding schema build via
  ``aegir.rl.decoding.composition_json_schema``.
- Policy-load dry run (no actual weights touched).

Run::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/p5_scaffold_smoke.py
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.verifier import CompositionEntry  # noqa: E402
from aegir.rl.checkpointing import (  # noqa: E402
    CheckpointConfig,
    build_run_metadata,
    hash_locked_weights,
    latest_checkpoint,
    read_metadata,
    verify_resume_metadata,
    write_metadata,
)
from aegir.rl.decoding import (  # noqa: E402
    DecodingConfig,
    schema_dry_run,
)
from aegir.rl.grpo_loop import (  # noqa: E402
    GRPOConfig,
    compute_group_advantages,
    grpo_iteration,
    hot_reload_catalog,
    init_grpo_state,
    load_metrics_history,
    score_composition,
)
from aegir.rl.policy import PolicyConfig, policy_load_dry_run  # noqa: E402
from aegir.rl.sae_logging import SAELogConfig, SAELogRecord, SAELogger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("p5-scaffold-smoke")


SDG_CLASSES = [
    "sdg:LabRun", "sdg:Trace", "sdg:Sample", "sdg:Reading",
    "sdg:HipaaRule", "sdg:ColumnPolicy", "sdg:Audit",
    "sdg:eBPFProgram", "sdg:KernelHook", "sdg:Syscall", "sdg:Map",
    "sdg:LineageEdge", "sdg:Transformation", "sdg:Allocation",
    "sdg:MassFunction", "sdg:BeliefInterval", "sdg:Evidence", "sdg:Claim",
]


def stub_policy_rollout(catalog, group_size: int):
    """Stub policy emitting a mix of good and bad compositions.
    Stands in for the real Qwen rollout."""
    rng = random.Random(42)
    out = []
    for _ in range(group_size):
        n_templates = rng.randint(3, 8)
        comp = []
        for _ in range(n_templates):
            tmpl = rng.choice(catalog.templates)
            fillers = {
                slot_name: rng.choice(SDG_CLASSES)
                for slot_name in tmpl.slot_types.keys()
            }
            comp.append(CompositionEntry(
                template_id=tmpl.template_id,
                slot_fillers=fillers,
            ))
        out.append(comp)
    return out


def stub_policy_step(compositions, advantages):
    """Stub gradient step. Logs but does nothing."""
    logger.info(
        "stub policy step: %d compositions, mean advantage = %.4f",
        len(compositions),
        sum(advantages) / max(len(advantages), 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iterations", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("P5 scaffold smoke test")
    print("=" * 70)

    # 1. catalog hot-reload
    print("\n[1/5] catalog load + hot-reload")
    cfg = GRPOConfig()
    state = init_grpo_state(cfg)
    print(f"  catalog: {len(state.catalog.templates)} templates "
          f"(version={state.catalog.version})")
    changed = hot_reload_catalog(state)
    print(f"  hot-reload no-op detected unchanged: {not changed}")

    # 2. policy-load dry run
    print("\n[2/5] policy-load dry run")
    policy_cfg = PolicyConfig()
    plan = policy_load_dry_run(policy_cfg)
    for k, v in plan.items():
        print(f"  {k}: {v}")

    # 3. constrained-decode schema
    print("\n[3/5] constrained-decode schema")
    decoding_cfg = DecodingConfig()
    print(f"  backend: {decoding_cfg.backend}")
    summary = schema_dry_run(state.catalog)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 4. single composition score
    print("\n[4/5] single-composition score (locked verifier)")
    rng = random.Random(0)
    sample = []
    for _ in range(4):
        tmpl = rng.choice(state.catalog.templates)
        sample.append(CompositionEntry(
            template_id=tmpl.template_id,
            slot_fillers={s: rng.choice(SDG_CLASSES) for s in tmpl.slot_types.keys()},
        ))
    res = score_composition(sample, state, iteration=999)  # warmup window already past
    print(f"  R={res.R:.4f}  R_A={res.R_A:.2f}  R_B={res.R_B:.2f}  "
          f"R_C={res.R_C:.3f}  R_D={res.R_D:.3f}")

    # 5. GRPO iteration with stub policy
    print(f"\n[5/5] {args.n_iterations} GRPO iterations against stub policy")
    t0 = time.time()
    for i in range(args.n_iterations):
        m = grpo_iteration(
            state, iteration=i,
            rollout_fn=stub_policy_rollout,
            policy_step_fn=stub_policy_step,
        )
        print(f"  iter {i:>2d}: R mean={m.rewards_mean:.4f} std={m.rewards_std:.4f} "
              f"adv std={m.advantage_std:.4f} R_A pass={m.r_a_pass_rate:.0%} "
              f"invalid={m.n_invalid_compositions}")
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.2f}s "
          f"({elapsed / args.n_iterations:.2f}s / iteration)")

    # advantage normalization sanity
    sample_rewards = [m.rewards_mean for m in state.metrics]
    if len(sample_rewards) >= 2:
        advs = compute_group_advantages(sample_rewards)
        adv_mean = sum(advs) / len(advs)
        adv_std = (sum((a - adv_mean) ** 2 for a in advs) / len(advs)) ** 0.5
        print(f"\nadvantage normalization sanity: mean={adv_mean:.4f}  std={adv_std:.4f}")

    # 6. checkpoint round-trip (metadata sidecar, SAE spill/load, metrics JSONL)
    print("\n[6/6] checkpoint round-trip")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # metadata sidecar
        metadata = build_run_metadata(
            catalog_path=cfg.catalog_path,
            t_i_cache_path=cfg.t_i_cache_path,
            null_stats_path=cfg.null_stats_path,
        )
        ckpt_dir = td_path / "checkpoint-50"
        write_metadata(metadata, ckpt_dir)
        loaded = read_metadata(ckpt_dir)
        assert loaded.run_id == metadata.run_id
        assert loaded.locked_weights_hash == hash_locked_weights()
        print(f"  metadata sidecar: run_id={metadata.run_id[:8]}…  "
              f"catalog_v={metadata.catalog_version}  "
              f"locked_w_hash={metadata.locked_weights_hash}")

        # verify_resume — happy path
        verify_resume_metadata(
            ckpt_dir,
            catalog_path=cfg.catalog_path,
            t_i_cache_path=cfg.t_i_cache_path,
            null_stats_path=cfg.null_stats_path,
            strict=True,
        )
        print("  verify_resume_metadata (no drift): ✓")

        # SAE spill + load round-trip
        sae_logger = SAELogger(cfg=SAELogConfig())
        sae_logger.records = [
            SAELogRecord(step=1, token_index=0,
                         top_feature_indices=[42, 7, 19],
                         top_feature_activations=[0.91, 0.66, 0.42],
                         reconstruction_loss=0.0123),
            SAELogRecord(step=1, token_index=4,
                         top_feature_indices=[42, 80211, 33],
                         top_feature_activations=[0.88, 0.71, 0.39],
                         reconstruction_loss=0.0117),
        ]
        sae_path = ckpt_dir / "sae_features.jsonl"
        n_written = sae_logger.spill_to_disk(sae_path)
        assert n_written == 2 and sae_logger.records == []
        sae_reloaded = SAELogger(cfg=SAELogConfig())
        n_loaded = sae_reloaded.load_from_disk(sae_path)
        assert n_loaded == 2 and len(sae_reloaded.records) == 2
        print(f"  SAE spill→load: {n_written} written, {n_loaded} reloaded")

        # GRPO metrics JSONL round-trip
        metrics_path = str(td_path / "grpo_metrics.jsonl")
        metrics_cfg = GRPOConfig(
            catalog_path=cfg.catalog_path,
            metrics_jsonl_path=metrics_path,
        )
        m_state = init_grpo_state(metrics_cfg)
        for i in range(3):
            grpo_iteration(m_state, iteration=i,
                           rollout_fn=stub_policy_rollout,
                           policy_step_fn=None)
        # simulate restart: discard m_state, reload metrics from disk
        history = load_metrics_history(metrics_path)
        assert len(history) == 3
        m_state2 = init_grpo_state(metrics_cfg)
        assert len(m_state2.metrics) == 3
        print(f"  GRPO metrics JSONL: {len(history)} iterations replayed across simulated restart")

        # latest_checkpoint discovery
        (td_path / "checkpoint-100").mkdir()
        (td_path / "checkpoint-150").mkdir()
        latest = latest_checkpoint(td_path)
        assert latest is not None and latest.name == "checkpoint-150"
        print(f"  latest_checkpoint discovery: {latest.name}")

        # CheckpointConfig sanity
        cp_cfg = CheckpointConfig(save_steps=50, save_total_limit=3)
        assert cp_cfg.save_strategy == "steps"
        print(f"  CheckpointConfig defaults: save_steps={cp_cfg.save_steps} "
              f"save_total_limit={cp_cfg.save_total_limit}")

    print("\nP5 scaffold smoke test ✓ PASS")
    print("  catalog hot-reload:        ✓")
    print("  policy-load dry run:       ✓")
    print("  decoding-schema build:     ✓")
    print("  verifier scoring:          ✓")
    print("  GRPO iteration end-to-end: ✓")
    print("  checkpoint round-trip:     ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
