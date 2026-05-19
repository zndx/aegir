#!/usr/bin/env python
"""Family simplex analysis — which family combinations integrate well?

Each chapter is a simplex in the family-complex:

  - 0-simplex:  {fam_A}                        (single-family chapter)
  - 1-simplex:  {fam_A, fam_B}                 (chapter spans two families)
  - 2-simplex:  {fam_A, fam_B, fam_C}          (triangle)
  - 3-simplex:  {fam_A, fam_B, fam_C, fam_D}   (tetrahedron)

R_axiom (and the other R-scorers) act as a "weight" on each simplex.
The face map ∂ : C_k → C_{k-1} sends a k-simplex to its k+1 (k-1)-faces.
If R_axiom collapses on a k-simplex but holds on (k-1)-faces, the
incompatibility is induced by adding the k-th vertex — pinpoint the
combination that doesn't compose.

The output is a per-simplex efficacy table + a face-map delta table
that says, for each underperforming k-simplex, which face produces
the largest delta if we drop one family.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/family_simplex_analysis.py \\
        --chapters-run /raid/.../topic_first_v0/<id>/
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("simplex")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-run", required=True,
                   help="Path to a chapters run dir (has chapters.parquet + "
                        "verification.parquet)")
    p.add_argument("--r-axiom-floor", type=float, default=0.45,
                   help="Per-simplex R_axiom floor for 'composes well' "
                        "classification")
    p.add_argument("--min-count", type=int, default=2,
                   help="Minimum chapters per simplex to report (statistical "
                        "noise floor)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()
    run_dir = Path(args.chapters_run)

    chapters = pq.read_table(run_dir / "chapters.parquet").to_pandas()
    verif = pq.read_table(run_dir / "verification.parquet").to_pandas()
    df = chapters.merge(verif, on="chapter_id", suffixes=("", "_v"))
    logger.info(f"loaded {len(df)} chapters with verification scores from {run_dir.name}")

    # Build per-chapter family simplex (frozenset, sorted-tuple for keying)
    df["simplex"] = df["template_families"].apply(
        lambda fs: tuple(sorted(set(fs))) if fs is not None else ()
    )
    df["dim"] = df["simplex"].apply(lambda s: len(s) - 1)

    # --- Per-simplex efficacy table ---------------------------------------
    by_simplex: dict[tuple, dict] = {}
    for simplex, sub in df.groupby("simplex"):
        if len(sub) < args.min_count:
            continue
        by_simplex[simplex] = {
            "dim": len(simplex) - 1,
            "n": len(sub),
            "r_topic": float(sub.r_topic.mean()),
            "r_iri": float(sub.r_iri.mean()),
            "r_density": float(sub.r_density.mean()),
            "r_axiom": float(sub.r_axiom.mean()),
            "r_axiom_std": float(sub.r_axiom.std() or 0.0),
            "r_comp": float(sub.r_composite.mean()),
            "accepted_frac": float((sub.status == "accepted").mean()),
        }

    print()
    print("=== Per-simplex efficacy (n ≥ "
          f"{args.min_count}; sorted by dim then n) ===")
    print()
    print(f"{'dim':>3} {'n':>4} {'R_axiom':>9} {'±σ':>6} {'R_iri':>7} "
          f"{'R_dens':>7} {'accept':>8} {'family simplex':<60}")
    print("─" * 120)
    for simplex, s in sorted(by_simplex.items(),
                              key=lambda kv: (kv[1]["dim"], -kv[1]["n"])):
        fam_short = ",".join(f.split("_", 1)[-1][:10] for f in simplex)
        gate = "✓" if s["r_axiom"] >= args.r_axiom_floor else "✗"
        print(f"{s['dim']:>3} {s['n']:>4} {s['r_axiom']:>8.3f}{gate} "
              f"{s['r_axiom_std']:>6.3f} {s['r_iri']:>7.3f} "
              f"{s['r_density']:>7.3f} {s['accepted_frac']:>7.0%} "
              f"  {fam_short}")

    # --- Face-map delta analysis ------------------------------------------
    # For each k-simplex with R_axiom below floor, look at its (k-1)-faces.
    # The "drop-which-family" delta = R_axiom(face) - R_axiom(simplex).
    # Largest delta → that face is the strongest sub-combination.
    print()
    print("=== Face-map deltas (which family to DROP from underperforming k-simplices) ===")
    print()
    print(f"{'k-simplex':<55} {'R_ax':>6}  {'drop':>15}  {'→ face R_ax':>11}  {'Δ':>6}")
    print("─" * 110)

    failing = [(s, info) for s, info in by_simplex.items()
               if info["r_axiom"] < args.r_axiom_floor and info["dim"] >= 1]
    failing.sort(key=lambda kv: (kv[1]["dim"], kv[1]["r_axiom"]))

    if not failing:
        print("  (no k-simplices below floor with n ≥ min-count)")
    for simplex, info in failing:
        # Face is simplex with one vertex removed
        face_results = []
        for f in simplex:
            face = tuple(x for x in simplex if x != f)
            face_info = by_simplex.get(face)
            face_r_axiom = face_info["r_axiom"] if face_info else float("nan")
            face_results.append((f, face_r_axiom, face_info))
        face_results.sort(key=lambda x: -(x[1] if not np.isnan(x[1]) else -1))
        best_drop_f, best_face_r, best_face_info = face_results[0]

        simplex_short = ",".join(f.split("_", 1)[-1][:10] for f in simplex)
        drop_short = best_drop_f.split("_", 1)[-1][:14]
        if best_face_info is None:
            face_note = "(face n<min)"
            delta_str = "—"
        else:
            face_note = f"{best_face_r:.3f}"
            delta_str = f"+{best_face_r - info['r_axiom']:.3f}"
        print(f"{simplex_short:<55} {info['r_axiom']:>6.3f}  {drop_short:>15}  "
              f"{face_note:>11}  {delta_str:>6}")

    # --- Pair-level "incompatibility" summary -----------------------------
    # For each unordered pair of families, average R_axiom across chapters
    # that contain BOTH families (regardless of dim). Compare to chapters
    # that contain EITHER alone (or neither + same dim baseline).
    print()
    print("=== Pair affinity (R_axiom mean for chapters containing both) ===")
    print()

    all_families: set[str] = set()
    for s in df["simplex"]:
        all_families.update(s)

    pair_stats: list[tuple] = []
    for a, b in combinations(sorted(all_families), 2):
        mask = df["simplex"].apply(lambda s: a in s and b in s)
        sub = df[mask]
        if len(sub) < args.min_count:
            continue
        # Comparison: chapters with a but not b, or b but not a
        solo_a_mask = df["simplex"].apply(lambda s: a in s and b not in s)
        solo_b_mask = df["simplex"].apply(lambda s: b in s and a not in s)
        solo_a = df[solo_a_mask]
        solo_b = df[solo_b_mask]
        solo_axiom = []
        if len(solo_a) > 0:
            solo_axiom.append(solo_a.r_axiom.mean())
        if len(solo_b) > 0:
            solo_axiom.append(solo_b.r_axiom.mean())
        solo_mean = float(np.mean(solo_axiom)) if solo_axiom else float("nan")
        pair_stats.append((a, b, len(sub), float(sub.r_axiom.mean()),
                           solo_mean))

    # Sort by pair R_axiom ascending so the worst pairings surface first
    pair_stats.sort(key=lambda r: r[3])
    print(f"{'family A':<30} {'family B':<30} {'n':>3} "
          f"{'pair_R_ax':>10} {'solo_R_ax':>10} {'Δ vs solo':>10}")
    print("─" * 110)
    for a, b, n, pair_r, solo_r in pair_stats:
        delta = pair_r - solo_r if not np.isnan(solo_r) else float("nan")
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "—"
        a_short = a.split("_", 1)[-1][:28]
        b_short = b.split("_", 1)[-1][:28]
        print(f"{a_short:<30} {b_short:<30} {n:>3} "
              f"{pair_r:>10.3f} {solo_r:>10.3f} {delta_str:>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
