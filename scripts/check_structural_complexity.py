#!/usr/bin/env python
"""Structural-complexity KPI for a realized DDL spine — is the schema corpus as structurally DIVERSE
and rich as real-world relational systems, or flat boilerplate?

Reads ``structural_complexity.parquet`` (emitted by ``build_ddl_spine.py --realize``) and reports the
growth + diversity dynamics: the per-template table multiplier (vs the flat one-table baseline), the
structural-profile mix + its entropy, and the EAV-ratio / M:N-density / FK-depth distributions. It checks
these against PROVISIONAL diversity TARGETS calibrated to public observations of real schema styles —
Magento-class EAV (entity + per-datatype value tables), Odoo-class many-to-many junctions, Kimball
star/snowflake. (Those are *measurements of structural distributions* — facts — not the projects'
copyrighted schemas; the generator is clean-room, see memory cleanroom_ddl_generation.)

    uv run --no-sync python scripts/check_structural_complexity.py --spine-run /tmp/sl_realize/<run>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# PROVISIONAL diversity targets (ratchet). Each schema corpus should EXHIBIT the spectrum real systems
# span: flat↔EAV, dense M:N, dimensional. Targets are bands, not the destination (north-star = genuine
# real-world complexity; provisional_scaffolding_not_goals).
TARGETS = {
    "min_profiles_used": 4,            # ≥4 of {normalized,eav,junction,star,snowflake}
    "min_tables_per_template": 2.5,    # super-linear vs the flat (1.0) baseline
    "min_eav_share": 0.10,             # fraction of templates realized as EAV
    "min_junction_share": 0.10,        # fraction with a many-to-many association class
    "min_dimensional_share": 0.05,     # fraction realized as star/snowflake
    "min_max_fk_depth": 2,             # multi-hop joins exist (the model/Atelier must learn them)
}


def _entropy(counts) -> float:
    tot = sum(counts)
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0) if tot else 0.0


def score(run: Path) -> dict:
    rows = pq.read_table(run / "structural_complexity.parquet").to_pylist()
    n = len(rows)
    profiles = Counter(r["profile"] for r in rows)
    tpt = [r["n_tables"] for r in rows]
    eav = sum(1 for r in rows if r["profile"] == "eav")
    junc = sum(1 for r in rows if (r.get("junction_tables") or 0) > 0)
    dim = sum(1 for r in rows if r["profile"] in ("star", "snowflake"))
    return {
        "run": str(run), "n_templates": n,
        "profiles_used": len(profiles), "profile_distribution": dict(profiles),
        "profile_entropy_bits": round(_entropy(list(profiles.values())), 3),
        "tables_per_template": {"mean": round(st.mean(tpt), 2), "min": min(tpt), "max": max(tpt),
                                "p50": int(st.median(tpt)), "total": sum(tpt)},
        "growth_multiplier_vs_flat": round(sum(tpt) / n, 2),
        "total_views": sum(r["n_views"] for r in rows),
        "eav_share": round(eav / n, 3), "junction_share": round(junc / n, 3),
        "dimensional_share": round(dim / n, 3),
        "mean_eav_ratio_when_eav": round(st.mean([r["eav_ratio"] for r in rows if r["profile"] == "eav"] or [0]), 3),
        "mean_m2n_density": round(st.mean([r["m2n_density"] for r in rows]), 3),
        "max_fk_depth": max(r["fk_depth"] for r in rows),
        "mean_attr_count": round(st.mean([r["attr_count"] for r in rows]), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine-run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.spine_run)
    if not (run / "structural_complexity.parquet").exists():
        print(f"no structural_complexity.parquet in {run} (build with --realize)", file=sys.stderr)
        return 2
    rep = score(run)
    checks = {
        "profiles_used": (rep["profiles_used"], ">=", TARGETS["min_profiles_used"]),
        "tables_per_template": (rep["tables_per_template"]["mean"], ">=", TARGETS["min_tables_per_template"]),
        "eav_share": (rep["eav_share"], ">=", TARGETS["min_eav_share"]),
        "junction_share": (rep["junction_share"], ">=", TARGETS["min_junction_share"]),
        "dimensional_share": (rep["dimensional_share"], ">=", TARGETS["min_dimensional_share"]),
        "max_fk_depth": (rep["max_fk_depth"], ">=", TARGETS["min_max_fk_depth"]),
    }
    rep["checks"] = {k: {"value": v, "op": op, "target": t, "pass": v >= t} for k, (v, op, t) in checks.items()}
    rep["all_pass"] = all(c["pass"] for c in rep["checks"].values())
    out = Path(args.out) if args.out else run / "structural_complexity_report.json"
    out.write_text(json.dumps(rep, indent=2))

    print(f"STRUCTURAL COMPLEXITY — {run.name}  ({rep['n_templates']} templates → {rep['tables_per_template']['total']} tables)")
    print(f"  growth multiplier vs flat : {rep['growth_multiplier_vs_flat']}×  · views {rep['total_views']}")
    print(f"  profiles: {rep['profile_distribution']}  (entropy {rep['profile_entropy_bits']} bits)")
    print(f"  EAV share {rep['eav_share']} (ratio {rep['mean_eav_ratio_when_eav']}) · junction share "
          f"{rep['junction_share']} · dimensional {rep['dimensional_share']} · max FK depth {rep['max_fk_depth']}")
    for k, c in rep["checks"].items():
        print(f"    {'✓' if c['pass'] else '✘'} {k:22s} {c['value']!s:>6} {c['op']} {c['target']}")
    print(f"\n  KPI: {'🟢 diverse' if rep['all_pass'] else '🔴 under-diverse'}  → {out}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
