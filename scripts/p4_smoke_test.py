"""p4_smoke_test — validate reward-signal propagation through a single
GRPO group iteration over the locked SDG verifier.

Per the v0.5 concept brief P4 — this is the **minimal** smoke test:
no policy training, no model weights updated. The goal is to
demonstrate that:

1. A stub policy can emit ``N`` candidate compositions over the
   combined catalog.
2. Each composition can be rendered and scored end-to-end via
   the locked verifier ``R(O, I)`` at locked weights
   ``{0.50, 0.05, 0.45}``.
3. The resulting rewards form a non-degenerate GRPO group:
   group mean and standard deviation are recorded, group-
   relative advantages computed.
4. The reward signal is deterministic across runs: identical
   stub policy outputs produce identical group-relative
   advantages.
5. The reward signal differentiates: compositions that exercise
   slot-bearing complex axioms score higher than compositions
   of trivial single-axiom templates.

This validates the reward-side wiring required by GRPO (or any
policy-gradient RL family). Full GRPO training with the
``SAE-Res-Qwen3.5-27B-W80K-L0_100`` policy is out of scope here;
the P5 phase commits ~200 GPU-hours to that.

Usage:
    bash scripts/setup_jvm_env.sh
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/p4_smoke_test.py
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aegir.ontology.schema import Catalog, load_catalog  # noqa: E402
from aegir.ontology.verifier import (  # noqa: E402
    CompositionEntry,
    W_R_B,
    W_R_C,
    W_R_D,
    verify,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

DEFAULT_CATALOG = REPO_ROOT / "src" / "aegir" / "ontology" / "catalog" / "combined.json"
DEFAULT_T_I_CACHE = REPO_ROOT / "src" / "aegir" / "ontology" / "T_I.pkl"
DEFAULT_NULL_STATS = REPO_ROOT / "src" / "aegir" / "ontology" / "null_stats.json"

GROUP_SIZE = 8


@dataclass
class GroupResult:
    """Outcome of a single GRPO group iteration."""

    rewards: list[float] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


def stub_policy(catalog: Catalog, group_size: int, rng: random.Random) -> list[tuple[str, list[CompositionEntry]]]:
    """Emit ``group_size`` candidate compositions of varying quality.

    Stand-in for the GRPO policy's sampling step — no model
    forward pass, no gradient. Each sample carries a short
    description for diagnostic readout.

    Sample profiles:

    - Profile A — high-quality slot-rich composition: 12 randomly-
      sampled is_complex-True templates with diverse slot fillers.
    - Profile B — medium-quality mixed composition: 8 templates
      drawn from the verbalized pool with a mix of complex and
      basic shapes.
    - Profile C — low-quality trivial composition: 4 basic
      F1-subclass templates, all with the same filler.
    - Profile D — degenerate empty-fill composition: 6 templates
      with trivially-named ``sdg:Stub`` fillers.

    The stub generates 2 of each profile to fill an 8-sample group.
    """
    pool_complex = [t for t in catalog.templates if t.is_complex and t.verbal_template]
    pool_verb = [t for t in catalog.templates if t.verbal_template]
    pool_basic = [t for t in catalog.templates
                  if t.provenance.get("family") == "F1-subclass"
                  and t.verbal_template]

    samples: list[tuple[str, list[CompositionEntry]]] = []

    realistic_fillers_pool = [
        "sdg:LabRun", "sdg:BloodSample", "sdg:Spectrometer", "sdg:HplcInstrument",
        "sdg:SpectralReading", "sdg:Wavelength", "sdg:Nanometer", "sdg:Celsius",
        "sdg:DataTable", "sdg:Column", "sdg:Constraint", "sdg:DataRetentionPolicy",
        "sdg:HttpRequestSpan", "sdg:CheckoutService", "sdg:KubernetesPod",
        "sdg:NetworkPacketEvent", "sdg:XdpProgram", "sdg:OpenSyscall",
        "sdg:OutlierClaim", "sdg:AttributeSet", "sdg:Lift", "sdg:Aggregation",
        "sdg:RequirementId", "sdg:ControlOwner", "sdg:AuditFinding", "sdg:Tier",
    ]

    def composition_a(rng: random.Random) -> list[CompositionEntry]:
        chosen = rng.sample(pool_complex, min(12, len(pool_complex)))
        return [
            CompositionEntry(
                template_id=t.template_id,
                slot_fillers={
                    name: rng.choice(realistic_fillers_pool)
                    for name in t.slot_types
                },
            )
            for t in chosen
        ]

    def composition_b(rng: random.Random) -> list[CompositionEntry]:
        chosen = rng.sample(pool_verb, min(8, len(pool_verb)))
        return [
            CompositionEntry(
                template_id=t.template_id,
                slot_fillers={
                    name: rng.choice(realistic_fillers_pool)
                    for name in t.slot_types
                },
            )
            for t in chosen
        ]

    def composition_c(rng: random.Random) -> list[CompositionEntry]:
        chosen = rng.sample(pool_basic, min(4, len(pool_basic)))
        same_filler = rng.choice(realistic_fillers_pool)
        return [
            CompositionEntry(
                template_id=t.template_id,
                slot_fillers={name: same_filler for name in t.slot_types},
            )
            for t in chosen
        ]

    def composition_d(rng: random.Random) -> list[CompositionEntry]:
        chosen = rng.sample(pool_verb, min(6, len(pool_verb)))
        return [
            CompositionEntry(
                template_id=t.template_id,
                slot_fillers={name: "sdg:Stub" for name in t.slot_types},
            )
            for t in chosen
        ]

    profile_makers = [
        ("A_complex_slot_rich",   composition_a),
        ("A_complex_slot_rich_2", composition_a),
        ("B_mixed",               composition_b),
        ("B_mixed_2",             composition_b),
        ("C_trivial_basic",       composition_c),
        ("C_trivial_basic_2",     composition_c),
        ("D_stub_filler",         composition_d),
        ("D_stub_filler_2",       composition_d),
    ]
    for label, maker in profile_makers[:group_size]:
        samples.append((label, maker(rng)))
    return samples


def score_group(
    catalog: Catalog,
    samples: list[tuple[str, list[CompositionEntry]]],
    t_i_cache: Path,
    null_stats: Path,
) -> GroupResult:
    """Score a group of compositions; compute GRPO group-relative advantages."""
    rewards: list[float] = []
    comps: list[dict] = []
    descriptions: list[str] = []
    for label, comp in samples:
        result = verify(
            comp, catalog,
            t_i_cache_path=t_i_cache,
            null_stats_path=null_stats,
        )
        rewards.append(result.R)
        comps.append({
            "R_A": result.R_A, "R_B": result.R_B,
            "R_C": result.R_C, "R_D": result.R_D,
            "n_entries": len(comp),
        })
        descriptions.append(label)

    rewards_arr = np.array(rewards)
    mean = float(rewards_arr.mean())
    std = float(rewards_arr.std())
    advantages = (rewards_arr - mean).tolist()
    if std > 1e-9:
        advantages = ((rewards_arr - mean) / std).tolist()

    return GroupResult(
        rewards=rewards,
        components=comps,
        advantages=advantages,
        descriptions=descriptions,
    )


def main() -> int:
    catalog = load_catalog(str(DEFAULT_CATALOG))
    logger.info("loaded combined catalog: %d templates", len(catalog.templates))

    print("=" * 80)
    print("P4 smoke test — single GRPO group iteration over the locked verifier")
    print("=" * 80)
    print(f"  catalog:    {DEFAULT_CATALOG}")
    print(f"  T_I cache:  {DEFAULT_T_I_CACHE}")
    print(f"  null_stats: {DEFAULT_NULL_STATS}")
    print(f"  weights:    a={W_R_B}, b={W_R_C}, c={W_R_D}")
    print(f"  group size: {GROUP_SIZE}")
    print()

    rng_seed = 4649
    rng = random.Random(rng_seed)
    samples = stub_policy(catalog, GROUP_SIZE, rng)
    print(f"=== sampling {GROUP_SIZE} compositions from stub policy (seed={rng_seed}) ===")
    for label, comp in samples:
        print(f"  {label:25s}  n_entries={len(comp)}")
    print()

    print("=== scoring group via locked R(O, I) ===")
    result = score_group(catalog, samples, DEFAULT_T_I_CACHE, DEFAULT_NULL_STATS)
    print()

    rewards_arr = np.array(result.rewards)
    advantages_arr = np.array(result.advantages)
    print(f"=== group statistics ===")
    print(f"  mean reward:         {rewards_arr.mean():.6f}")
    print(f"  std reward:          {rewards_arr.std():.6f}")
    print(f"  min reward:          {rewards_arr.min():.6f}  ({result.descriptions[int(rewards_arr.argmin())]})")
    print(f"  max reward:          {rewards_arr.max():.6f}  ({result.descriptions[int(rewards_arr.argmax())]})")
    print(f"  reward range:        {rewards_arr.max() - rewards_arr.min():.6f}")
    print(f"  advantage mean:      {advantages_arr.mean():.6f}  (should be ~0)")
    print(f"  advantage std:       {advantages_arr.std():.6f}  (should be ~1 after z-norm)")
    print()

    print("=== per-sample breakdown ===")
    print(f"  {'description':28s} {'R':>9s}  {'R_A':>5s}  {'R_B':>5s}  {'R_C':>5s}  {'R_D':>5s}  {'advantage':>10s}")
    for desc, R, comp, adv in zip(result.descriptions, result.rewards, result.components, result.advantages):
        print(
            f"  {desc:28s} {R:>9.4f}  "
            f"{comp['R_A']:>5.2f}  {comp['R_B']:>5.2f}  {comp['R_C']:>5.2f}  {comp['R_D']:>5.2f}  "
            f"{adv:>10.4f}"
        )
    print()

    # Determinism check: re-run and confirm identical advantages.
    print("=== determinism check (re-run with same seed) ===")
    rng2 = random.Random(rng_seed)
    samples2 = stub_policy(catalog, GROUP_SIZE, rng2)
    result2 = score_group(catalog, samples2, DEFAULT_T_I_CACHE, DEFAULT_NULL_STATS)
    rewards_match = result.rewards == result2.rewards
    advantages_match = result.advantages == result2.advantages
    print(f"  rewards identical:    {'✓' if rewards_match else '✗'}")
    print(f"  advantages identical: {'✓' if advantages_match else '✗'}")
    print()

    # Discrimination check: high-quality (Profile A) should outscore
    # low-quality (Profile C / D) in expectation.
    print("=== discrimination check ===")
    profile_groups: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}
    for desc, R in zip(result.descriptions, result.rewards):
        kind = desc[0]
        profile_groups[kind].append(R)
    for kind in ["A", "B", "C", "D"]:
        if profile_groups[kind]:
            arr = np.array(profile_groups[kind])
            print(f"  Profile {kind}:  mean={arr.mean():.4f}  n={len(arr)}")
    print()

    # Exit gates per the brief P4:
    # - Trivial-reward training reaches mean R_trivial = 1.0 within 50 steps:
    #   Not applicable to this smoke test (no training); the analog is that
    #   the reward signal differentiates by quality profile in a single group.
    # - Per-step wall clock and memory profile within 2× of estimate:
    #   Score the group again with timing.
    import time
    rng3 = random.Random(rng_seed)
    samples3 = stub_policy(catalog, GROUP_SIZE, rng3)
    t0 = time.time()
    score_group(catalog, samples3, DEFAULT_T_I_CACHE, DEFAULT_NULL_STATS)
    t = time.time() - t0
    print(f"=== timing ===")
    print(f"  full group ({GROUP_SIZE} samples) end-to-end: {t:.2f}s")
    print(f"  per-sample: {t / GROUP_SIZE:.2f}s")
    print()

    all_gates_pass = (
        rewards_match
        and advantages_match
        and rewards_arr.std() > 0
        and rewards_arr.max() > rewards_arr.min()
        and (
            np.mean(profile_groups["A"]) > np.mean(profile_groups["C"])
            if profile_groups["A"] and profile_groups["C"] else True
        )
    )
    print(f"P4 smoke test {'✓ PASS' if all_gates_pass else '✗ FAIL'}")
    print(f"  determinism:        {'✓' if rewards_match and advantages_match else '✗'}")
    print(f"  non-degenerate std: {'✓' if rewards_arr.std() > 0 else '✗'}")
    print(f"  reward variance:    {'✓' if rewards_arr.max() > rewards_arr.min() else '✗'}")
    print(f"  quality-discrimination Profile A > Profile C: "
          f"{'✓' if np.mean(profile_groups['A']) > np.mean(profile_groups['C']) else '✗'}")

    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
