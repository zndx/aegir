#!/usr/bin/env python
"""E1 re-derivation — $0 proxy screen (run BEFORE any GPU re-pretrain).

E1 found the verifier composite cannot CI-cleanly discriminate downstream value
WITHIN a model — diagnosed as a compressed within-model R-range. Before spending
GPU re-splitting by a NEW proxy candidate (E6-A's S1->S3 transfer), screen whether
that candidate actually has more within-model discriminative spread than the
composite, and whether it carries information the composite lacks.

Joins E6-A per-chapter transfers (per_chapter.json) with verify_chapters'
per-chapter r_composite / r_topic / r_axiom. Reports, within each model:
  - normalized spread (IQR/median, std/mean, p90-p10) of each candidate proxy
  - Spearman(S1->S3, r_composite) and Spearman(S1->S3, r_topic)
A proxy worth re-pretraining on has spread COMPARABLE-OR-WIDER than the composite
AND is not redundant with it (|rho| well below 1). Deterministic; no GPU/LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = ["811408b392859708", "d7646714bdd5e16f"]
CORPUS_DIR = "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 1e-12 else float("nan")


def spread(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "cv": float(x.std() / x.mean()) if x.mean() else float("nan"),
            "iqr_over_med": float((q3 - q1) / p50) if p50 else float("nan"),
            "p90_p10": float(p90 - p10)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-chapter", default="/raid/checkpoints/aegir-artifacts/evidence/e6a/per_chapter.json")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    per = {r["chapter_id"]: r for r in json.load(open(args.per_chapter))}
    ver: dict[str, dict] = {}
    for run in RUNS:
        p = f"{CORPUS_DIR}/{run}/verification.parquet"
        try:
            for r in pq.read_table(p, columns=["chapter_id", "model", "r_composite",
                                               "r_topic", "r_axiom"]).to_pylist():
                ver[r["chapter_id"]] = r
        except Exception as e:
            print(f"  (no verification for {run}: {e})")

    joined = [{**per[cid], **ver[cid]} for cid in per if cid in ver]
    print(f"joined {len(joined)} chapters (per={len(per)}, verified={len(ver)})\n")

    proxies = ["s1s3", "s0s3", "r_composite", "r_topic", "r_axiom"]
    for model in ["cerebras/zai-glm-4.7", "xai/grok-4.3", "ALL"]:
        sub = joined if model == "ALL" else [j for j in joined if j["model"] == model]
        if len(sub) < 5:
            continue
        print(f"=== {model} (n={len(sub)}) ===")
        print(f"  {'proxy':<12} {'mean':>7} {'std':>7} {'cv':>7} {'iqr/med':>8} {'p90-p10':>8}")
        sp = {}
        for px in proxies:
            vals = np.array([j[px] for j in sub], float)
            sp[px] = vals
            s = spread(vals)
            print(f"  {px:<12} {s['mean']:7.3f} {s['std']:7.3f} {s['cv']:7.3f} "
                  f"{s['iqr_over_med']:8.3f} {s['p90_p10']:8.3f}")
        rho_c = spearman(sp["s1s3"], sp["r_composite"])
        rho_t = spearman(sp["s1s3"], sp["r_topic"])
        print(f"  Spearman(S1->S3, r_composite) = {rho_c:+.3f}   "
              f"Spearman(S1->S3, r_topic) = {rho_t:+.3f}")
        # verdict heuristic (within-model)
        if model != "ALL":
            cv_s = spread(sp["s1s3"])["cv"]; cv_c = spread(sp["r_composite"])["cv"]
            verdict = ("S1->S3 spread >= composite, low redundancy -> re-split candidate"
                       if cv_s >= cv_c * 0.8 and abs(rho_c) < 0.6
                       else "S1->S3 not clearly better within-model -> sharper design needed")
            print(f"  -> {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
