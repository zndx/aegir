#!/usr/bin/env python
"""Build/update src/aegir/ontology/family_complex.json from chapter runs.

Computes per-simplex efficacy from one or more chapter-run directories,
applies the inclusion rules (R_axiom ≥ floor & n ≥ min_count for
maximals; mean+std < floor & n ≥ min_count for punctures), and writes
the artifact. Idempotent: running on the same inputs produces the same
output (sort-stable).

Absorption: when both {A,B,C} and {A,B} pass the floor, only the larger
simplex is kept as a maximal — the smaller is implicitly allowed via
closure.

Usage::

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python \\
        scripts/build_family_complex.py \\
        --chapters-runs /raid/.../topic_first_v0/<id>/ \\
        --output src/aegir/ontology/family_complex.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.complex import FamilyComplex, SimplexEvidence  # noqa: E402

logger = logging.getLogger("build-family-complex")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chapters-runs", required=True, nargs="+")
    p.add_argument("--output", default="src/aegir/ontology/family_complex.json")
    p.add_argument("--r-axiom-floor", type=float, default=0.45)
    p.add_argument("--min-count", type=int, default=2)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    args = parse_args()

    # Aggregate chapter rows across all input runs
    all_rows: list[dict] = []
    source_run_ids: list[str] = []
    for run in args.chapters_runs:
        run = Path(run)
        chapters = pq.read_table(run / "chapters.parquet").to_pandas()
        verif = pq.read_table(run / "verification.parquet").to_pandas()
        df = chapters.merge(verif, on="chapter_id", suffixes=("", "_v"))
        df["simplex"] = df["template_families"].apply(
            lambda fs: frozenset(fs) if fs is not None else frozenset()
        )
        source_run_ids.append(run.name)
        for _, r in df.iterrows():
            all_rows.append({
                "simplex": r["simplex"],
                "r_topic": float(r.r_topic),
                "r_iri": float(r.r_iri),
                "r_density": float(r.r_density),
                "r_axiom": float(r.r_axiom),
                "r_composite": float(r.r_composite),
                "accepted": r.status == "accepted",
            })
    logger.info(f"aggregated {len(all_rows)} chapters from "
                f"{len(source_run_ids)} runs")

    # Group by simplex, compute statistics
    import collections
    import statistics

    by_simplex: dict[frozenset[str], list[dict]] = collections.defaultdict(list)
    for row in all_rows:
        by_simplex[row["simplex"]].append(row)

    candidates: list[SimplexEvidence] = []
    for simplex, rows in by_simplex.items():
        if len(rows) < args.min_count:
            continue
        r_ax_vals = [r["r_axiom"] for r in rows]
        candidates.append(SimplexEvidence(
            vertices=simplex,
            r_axiom_mean=statistics.mean(r_ax_vals),
            r_axiom_std=statistics.stdev(r_ax_vals) if len(r_ax_vals) > 1 else 0.0,
            r_iri_mean=statistics.mean(r["r_iri"] for r in rows),
            r_density_mean=statistics.mean(r["r_density"] for r in rows),
            accepted_frac=sum(1 for r in rows if r["accepted"]) / len(rows),
            n=len(rows),
        ))

    # Classify each candidate
    pass_floor = [c for c in candidates if c.r_axiom_mean >= args.r_axiom_floor]
    fail_confident = [c for c in candidates
                       if (c.r_axiom_mean + c.r_axiom_std) < args.r_axiom_floor]
    # candidates that are between (mean below floor but mean+std above) are
    # ambiguous — neither maximal nor confident-puncture. Skip.

    # Absorption: drop simplices that are proper subsets of larger passing ones
    pass_floor.sort(key=lambda e: -len(e.vertices))
    maximals: list[SimplexEvidence] = []
    for cand in pass_floor:
        if any(cand.vertices < m.vertices for m in maximals):
            continue  # absorbed
        maximals.append(cand)

    # Order outputs deterministically
    maximals.sort(key=lambda e: (-len(e.vertices), -e.r_axiom_mean, sorted(e.vertices)))
    fail_confident.sort(key=lambda e: (len(e.vertices), e.r_axiom_mean, sorted(e.vertices)))

    fc = FamilyComplex(
        maximal_simplices=maximals,
        measured_below_floor=fail_confident,
        r_axiom_floor=args.r_axiom_floor,
        min_count=args.min_count,
        source_runs=sorted(set(source_run_ids)),
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    fc.to_json(args.output)
    logger.info(f"wrote {args.output}")
    logger.info(f"  maximal simplices: {len(fc.maximal_simplices)}")
    for m in fc.maximal_simplices:
        logger.info(f"    {sorted(m.vertices)} (R_ax={m.r_axiom_mean:.3f}, n={m.n})")
    logger.info(f"  punctures: {len(fc.measured_below_floor)}")
    for p in fc.measured_below_floor:
        logger.info(f"    {sorted(p.vertices)} (R_ax={p.r_axiom_mean:.3f}, σ={p.r_axiom_std:.3f}, n={p.n})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
