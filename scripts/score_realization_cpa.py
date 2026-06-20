#!/usr/bin/env python
"""Score realization-CPA (#43): predicted template labels (HermiT realization) vs the held-out
reference, and the **selectivity** of the full Domain/Range TBox over the matched-token control.

Per the ontology-CPA-eval-methodology: report PR-style metrics (precision/recall/F1, not ROC-AUC,
not F1@0.5) and claim a win only when the selectivity delta's BCa bootstrap CI excludes 0 AND the
paired permutation p < 0.05. Realization recovers the RELATIONAL templates (those with object-property
restrictions) — so we also report recall against that recoverable subset (the fair denominator).

  uv run --no-sync python scripts/score_realization_cpa.py \
    --pred-full   evidence/realization_cpa/pred_full.parquet \
    --pred-control evidence/realization_cpa/pred_no-domain-range.parquet \
    --reference   /raid/.../atelier_release_v0_3/reference.parquet --out evidence/realization_cpa/score.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import abox  # noqa: E402
from aegir.ontology.ddl import parse_restrictions  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.utils.train import bca_ci, paired_permutation  # noqa: E402  (the #55 helpers — loop closed)


def _prf(pred: set, gt: set) -> tuple[float, float, float]:
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-full", required=True)
    ap.add_argument("--pred-control", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cat = load_catalog(Path(args.catalog))
    tindex = {abox._san(t.template_id): i for i, t in enumerate(cat.templates)}
    relational = {i for i, t in enumerate(cat.templates) if parse_restrictions(t)}  # recoverable subset

    # ground truth: chapter_id → set of template indices (over the reference's per-column rows)
    gt: dict[str, set] = {}
    for r in pq.read_table(args.reference, columns=["chapter_id", "template_id"]).to_pylist():
        idx = tindex.get(abox._san(r["template_id"]))
        if idx is not None:
            gt.setdefault(r["chapter_id"], set()).add(idx)

    def load_pred(path):
        return {r["chapter_id"]: set(r["pred_idx"]) for r in pq.read_table(path).to_pylist()}

    full, ctrl = load_pred(args.pred_full), load_pred(args.pred_control)
    chapters = [c for c in full if c in gt and c in ctrl]

    f1_full, f1_ctrl, deltas = [], [], []
    micro = {"tp_f": 0, "fp_f": 0, "fn_f": 0}
    rel_tp = rel_fn = 0  # recall vs the recoverable (relational) gt subset
    for c in chapters:
        g = gt[c]
        _, _, ff = _prf(full[c], g)
        _, _, fc = _prf(ctrl[c], g)
        f1_full.append(ff); f1_ctrl.append(fc); deltas.append(ff - fc)
        micro["tp_f"] += len(full[c] & g); micro["fp_f"] += len(full[c] - g); micro["fn_f"] += len(g - full[c])
        g_rel = g & relational
        rel_tp += len(full[c] & g_rel); rel_fn += len(g_rel - full[c])

    n = len(chapters)
    mp = micro["tp_f"] / (micro["tp_f"] + micro["fp_f"]) if (micro["tp_f"] + micro["fp_f"]) else 0.0
    mr = micro["tp_f"] / (micro["tp_f"] + micro["fn_f"]) if (micro["tp_f"] + micro["fn_f"]) else 0.0
    mf1 = (2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0
    rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) else 0.0
    mean_f1_full = sum(f1_full) / n if n else 0.0
    mean_f1_ctrl = sum(f1_ctrl) / n if n else 0.0
    sel = mean_f1_full - mean_f1_ctrl
    lo, hi = bca_ci(deltas) if n >= 2 else (sel, sel)
    pval = paired_permutation(f1_full, f1_ctrl, alternative="greater") if n else 1.0
    ci_clean = lo > 0.0 and pval < 0.05

    report = {
        "n_chapters": n,
        "micro": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf1, 4)},
        "relational_recall": round(rel_recall, 4),
        "macro_f1_full": round(mean_f1_full, 4), "macro_f1_control": round(mean_f1_ctrl, 4),
        "selectivity_f1": round(sel, 4), "selectivity_ci95": [round(lo, 4), round(hi, 4)],
        "permutation_p": round(pval, 5), "ci_clean": ci_clean,
        "verdict": ("DISCRIMINATING — realization-CPA selectivity > control, CI-clean (G-rel re-homed)"
                    if ci_clean else "NOT CI-clean — investigate"),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0 if ci_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
