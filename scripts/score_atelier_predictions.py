#!/usr/bin/env python
"""Score Atelier's blind column-classification predictions — the efficacy gate's scorer.

The handoff contract (settled in the sync advice, task #136.3): predictions arrive as parquet keyed
exactly like ``reference.parquet`` — required ``(table_id, column_id, predicted_code)``, optional
``(belief, plausibility)`` for cautious/calibrated scoring. The reference and the vocabulary
hierarchy (``parent_code``) are aegir-side; Atelier never sees the key.

THE SCORING RULE HONORS CALIBRATED COARSENESS (the ``rout_stop_address`` doctrine): full credit for
the minimal true set; hierarchical credit ``1/(1+d)`` along the ``parent_code`` chain in either
direction (an ancestor prediction is calibrated coarseness when the frame underdetermines the leaf;
a descendant is over-precision — both graded, never cliffed); unrelated codes score 0. Reference
codes may be SET-VALUED (``|``-separated) — the P5 references emit sets where the realize layer
knows a column genuinely spans leaves (junction FKs, multi-relation roles); credit is the max over
the set. When ``belief`` is present, Brier-style calibration is reported alongside accuracy — an
overconfident wrong singleton pays; honest mass on a coarse set does not.

Results are SLICED BY THE NAME-PROVENANCE LADDER (joined from ``corpus_columns.parquet``):
engine-derived / composed / structural-passthrough / degraded-mechanical — the provenance-sliced
ablation from the coordination plan, mechanical. Deterministic; no LLM / GPU / network.

    uv run --no-sync python scripts/score_atelier_predictions.py \\
        --predictions <their.parquet> --release build/atelier_release_preview --out build/atelier_score.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_hierarchy() -> "dict[str, str | None]":
    import pyarrow.parquet as pq
    recs = pq.read_table(REPO / "corpora/vocabulary/annotations.parquet").to_pylist()
    return {r["code"]: (r["parent_code"] or None) for r in recs}


def ancestry(code: str, parent: "dict[str, str | None]") -> "list[str]":
    """[code, parent, grandparent, …] — the chain the credit walks."""
    chain, seen = [], set()
    c: "str | None" = code
    while c and c not in seen:
        chain.append(c)
        seen.add(c)
        c = parent.get(c)
    return chain


def credit(pred: str, truth: str, parent: "dict[str, str | None]") -> float:
    """1/(1+d) along the parent chain in either direction; 0 when unrelated."""
    if not pred or not truth:
        return 0.0
    if pred == truth:
        return 1.0
    up_t = ancestry(truth, parent)
    if pred in up_t:                      # ancestor: calibrated coarseness
        return 1.0 / (1 + up_t.index(pred))
    up_p = ancestry(pred, parent)
    if truth in up_p:                     # descendant: over-precision
        return 1.0 / (1 + up_p.index(truth))
    return 0.0


def main() -> int:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="parquet keyed (table_id, column_id, predicted_code[, belief, plausibility])")
    ap.add_argument("--release", required=True,
                    help="the release dir (reference.parquet = key; corpus_columns.parquet = ladder slice)")
    ap.add_argument("--out", default="build/atelier_score.json")
    a = ap.parse_args()
    rel = Path(a.release)

    parent = load_hierarchy()
    # the key lives in the sibling <release>.key/ dir (key separation, 2026-07-03); legacy
    # releases with the reference beside the blind surface still score
    key = rel.parent / (rel.name + ".key") / "reference.parquet"
    if not key.exists():
        key = rel / "reference.parquet"
    ref = {(r["table_id"], r["column_id"]): r["reference_code"]
           for r in pq.read_table(key).to_pylist()}
    prov = {(r["table_id"], r["column_id"]): r.get("name_provenance", "?")
            for r in pq.read_table(rel / "corpus_columns.parquet").to_pylist()}
    preds = pq.read_table(a.predictions).to_pylist()

    per_rung: "dict[str, list[dict]]" = defaultdict(list)
    n_unkeyed = 0
    for p in preds:
        key = (p.get("table_id"), p.get("column_id"))
        truth_raw = ref.get(key)
        if truth_raw is None:
            n_unkeyed += 1
            continue
        truths = [t for t in str(truth_raw).split("|") if t]  # set-valued reference (P5+)
        pc = str(p.get("predicted_code") or "")
        c = max((credit(pc, t, parent) for t in truths), default=0.0)
        row = {"credit": c, "exact": float(pc in truths)}
        if p.get("belief") is not None:
            b = float(p["belief"])
            row["brier"] = (1 - b) ** 2 if c > 0 else b ** 2
        per_rung[prov.get(key, "?")].append(row)

    def agg(rows: "list[dict]") -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        out = {"n": n,
               "leaf_accuracy": round(sum(r["exact"] for r in rows) / n, 4),
               "hierarchical_score": round(sum(r["credit"] for r in rows) / n, 4),
               "coarse_credit_rate": round(sum(1 for r in rows if 0 < r["credit"] < 1) / n, 4),
               "miss_rate": round(sum(1 for r in rows if r["credit"] == 0) / n, 4)}
        briers = [r["brier"] for r in rows if "brier" in r]
        if briers:
            out["brier_calibration"] = round(sum(briers) / len(briers), 4)
        return out

    all_rows = [r for rows in per_rung.values() for r in rows]
    result = {
        "overall": agg(all_rows),
        "by_name_provenance": {rung: agg(rows) for rung, rows in sorted(per_rung.items())},
        "n_predictions": len(preds),
        "n_unkeyed": n_unkeyed,
        "n_reference": len(ref),
        "coverage": round((len(preds) - n_unkeyed) / len(ref), 4) if ref else 0.0,
        "release": str(rel),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
