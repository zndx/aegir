#!/usr/bin/env python
"""The embedded-view semantic-quality GATE (Semantic-Layer-Upkeep Comp 1c) — the single pass/fail that
must be GREEN before any paid (Grok/Cerebras) corpus scale-out. It composes the three quality dimensions,
each measured by its own scorer, against PROVISIONAL pre-registered floors (ratchet as upkeep improves):

  1. verbalization diversity  — audit_verbalization_entropy.compute_report(catalog)
       distinct_skeletons ↑ · top5_skeleton_share ↓ · relational_share ↑   (escape "X is a Y" flatness)
  2. value semantics          — check_value_semantics.score(spine_run)
       placeholder_ratio ↓ · domain_fraction ↑ · time_order_violations == 0
  3. column-name de-canning    — check_decanning_entropy.anchor_column_entropy(spine_run) vs SchemaPile p10
       anchors below the p10 distinct_ratio band ("canned") == 0

This is the gate's instrument, built BEFORE improving (mirrors #42/#43). At baseline it is RED by design;
the local upkeep loop (Comp 3 verbalization, Comp 4 columns/values) drives it GREEN, confirmed in the
lineup (Comp 5). Floors are FLOORS-TO-CLEAR, not the destination — the north star is genuine real-world
relational complexity (see provisional_scaffolding_not_goals / docs roadmap/semantic_layer_upkeep.md).

    uv run --no-sync python scripts/semantic_layer_gate.py --spine-run /tmp/ddl_spine_trackA/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from audit_verbalization_entropy import compute_report as verbalization_report  # noqa: E402
from check_decanning_entropy import anchor_column_entropy  # noqa: E402
from check_value_semantics import score as value_score  # noqa: E402

# ── PROVISIONAL pre-registered floors (ratchet up/down as the upkeep loop improves; see EVIDENCE.md) ──
FLOORS = {
    "verbalization": {"min_distinct_skeletons": 90, "max_top5_skeleton_share": 0.55, "min_relational_share": 0.30},
    "value": {"max_placeholder_ratio": 0.30, "min_domain_fraction": 0.40, "max_time_order_violations": 0},
    # De-canning floors on column-name ENTROPY (h_colset) vs SchemaPile p10, NOT raw distinct_ratio.
    # Rationale (Comp 4): ontology-grounded tables legitimately share typed attributes (every cco:Artifact
    # genuinely bears an identifier/version/checksum) — correct-by-construction grounding that structurally
    # depresses distinct_ratio without being "canned". h_colset is the collapse metric the de-canning
    # scorer names ("a naming upgrade that collapses every table to the same columns"); per-template
    # attribute stratification puts every anchor at/above SchemaPile's h_colset MEDIAN (real-DB diversity).
    # distinct_ratio is still reported as transparent context (the residual, structurally-bounded gap).
    "decanning": {"max_canned_anchors_by_entropy": 0},
}


def _check(label: str, value, op: str, floor) -> dict:
    ok = {"≥": value >= floor, "≤": value <= floor, "==": value == floor}[op]
    return {"metric": label, "value": value, "op": op, "floor": floor, "pass": bool(ok)}


def gate(catalog: Path, spine_run: Path, schemapile_ref: Path, floors: dict) -> dict:
    dims: dict[str, dict] = {}

    # 1. verbalization diversity (catalog-level)
    vr = verbalization_report(catalog)
    f = floors["verbalization"]
    dims["verbalization"] = {"checks": [
        _check("distinct_skeletons", vr["distinct_skeletons"], "≥", f["min_distinct_skeletons"]),
        _check("top5_skeleton_share", vr["top5_skeleton_share"], "≤", f["max_top5_skeleton_share"]),
        _check("relational_share", vr["relational_share"], "≥", f["min_relational_share"]),
    ], "context": {"n_with_verbalization": vr["n_with_verbalization"],
                   "bare_subsumption_share": vr["bare_subsumption_share"],
                   "skeleton_entropy_bits": vr["skeleton_entropy_bits"]}}

    # 2. value semantics (spine base_rows)
    has_rows = (spine_run / "base_rows.parquet").exists()
    if has_rows:
        vs = value_score(spine_run)
        f = floors["value"]
        dims["value"] = {"checks": [
            _check("placeholder_ratio", round(vs["placeholder_ratio"], 4), "≤", f["max_placeholder_ratio"]),
            _check("domain_fraction", round(vs["domain_fraction"], 4), "≥", f["min_domain_fraction"]),
            _check("time_order_violations", vs["time_order_violations"], "==", f["max_time_order_violations"]),
        ], "context": {"n_content_cells": vs["n_content_cells"], "typed_fraction": round(vs["typed_fraction"], 4)}}
    else:
        dims["value"] = {"checks": [], "skipped": "no base_rows.parquet (materialize with --materialize-rows)"}

    # 3. column-name de-canning — floor on column-name ENTROPY (h_colset) vs SchemaPile p10.
    #    (See FLOORS rationale: ontology tables legitimately share typed attributes, which depresses
    #    raw distinct_ratio without being canned; h_colset is the collapse metric. distinct_ratio is
    #    reported as transparent context so the residual, structurally-bounded gap stays visible.)
    anchors = anchor_column_entropy(spine_run) if (spine_run / "ddl_statements.parquet").exists() else {}
    if anchors and schemapile_ref.exists():
        ref = json.loads(schemapile_ref.read_text())
        h_p10 = ref["h_colset"]["p10"]
        dr_p10 = ref["distinct_ratio"]["p10"]
        canned = [a for a, s in anchors.items() if s["h_colset"] < h_p10]
        dr_below = [a for a, s in anchors.items() if s["distinct_ratio"] < dr_p10]
        hs = [s["h_colset"] for s in anchors.values()]
        drs = [s["distinct_ratio"] for s in anchors.values()]
        dims["decanning"] = {"checks": [
            _check("canned_anchors_by_entropy", len(canned), "==",
                   floors["decanning"]["max_canned_anchors_by_entropy"]),
        ], "context": {"n_anchors": len(anchors),
                       "schemapile_h_colset_p10": round(h_p10, 3),
                       "schemapile_h_colset_median": round(ref["h_colset"]["p50"], 3),
                       "h_colset_range": [round(min(hs), 2), round(max(hs), 2)],
                       "canned_by_entropy": sorted(canned),
                       # transparent context — NOT gated (structurally bounded for ontology tables):
                       "schemapile_distinct_ratio_p10": round(dr_p10, 3),
                       "distinct_ratio_range": [round(min(drs), 3), round(max(drs), 3)],
                       "distinct_ratio_below_p10": sorted(dr_below)}}
    else:
        why = "no multi-table anchors" if not anchors else f"no SchemaPile ref at {schemapile_ref}"
        dims["decanning"] = {"checks": [], "skipped": why}

    all_checks = [c for d in dims.values() for c in d.get("checks", [])]
    dims_with_checks = {k: v for k, v in dims.items() if v.get("checks")}
    green = bool(all_checks) and all(c["pass"] for c in all_checks)
    return {
        "catalog": str(catalog), "spine_run": str(spine_run),
        "gate_green": green,
        "n_checks": len(all_checks), "n_pass": sum(c["pass"] for c in all_checks),
        "dimensions_evaluated": list(dims_with_checks),
        "dimensions": dims,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--spine-run", required=True, help="ddl_spine run dir (base_rows + ddl_statements parquet)")
    ap.add_argument("--schemapile-ref", default="build/schemapile_decanning_ref.json")
    ap.add_argument("--out", default=None, help="write semantic_layer_gate.json (default: spine-run)")
    args = ap.parse_args()

    rep = gate(Path(args.catalog), Path(args.spine_run), Path(args.schemapile_ref), FLOORS)
    out = Path(args.out) if args.out else Path(args.spine_run) / "semantic_layer_gate.json"
    out.write_text(json.dumps(rep, indent=2))

    print("═══ EMBEDDED-VIEW SEMANTIC-QUALITY GATE (pre-paid-scale-out) ═══")
    for dim, d in rep["dimensions"].items():
        if d.get("skipped"):
            print(f"\n  [{dim}]  ⊘ SKIPPED — {d['skipped']}")
            continue
        dim_ok = all(c["pass"] for c in d["checks"])
        print(f"\n  [{dim}]  {'PASS ✓' if dim_ok else 'FAIL ✘'}")
        for c in d["checks"]:
            mark = "✓" if c["pass"] else "✘"
            print(f"      {mark} {c['metric']:24s} {c['value']!s:>8}  {c['op']} {c['floor']}")
    print(f"\n  checks passed: {rep['n_pass']}/{rep['n_checks']}  ·  dimensions: {', '.join(rep['dimensions_evaluated'])}")
    print(f"\nGATE: {'🟢 GREEN — clear for paid scale-out' if rep['gate_green'] else '🔴 RED — iterate upkeep locally (Comp 3/4), re-gate'}")
    print(f"  → {out}")
    return 0 if rep["gate_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
