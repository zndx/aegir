#!/usr/bin/env python
"""score_verbalizations — run the naturalness+faithfulness instruments over the catalog (RH step 4).

Perplexity (pythia-160m, GPU 4, offline) over EVERY template's frames vs its legacy DeepOnto string;
LLM-as-judge (engine, schema-constrained) on a seeded sample; a blinded human-preference sheet.
Artifact → build/verbalization_quality.json — consumed by semantic_layer_gate's quality dimension
(pre-registered floors). Run BEFORE and AFTER an intervention (e.g. the elaboration batch) to measure
lift against a frozen baseline (build/combined.pre-elab.json).

    uv run --no-sync python scripts/score_verbalizations.py --judge-sample 40
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import verbalization_quality as Q  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--judge-sample", type=int, default=40, help="templates judged by the engine (0 = skip)")
    ap.add_argument("--human-sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(REPO / "build/verbalization_quality.json"))
    ap.add_argument("--tag", default="", help="label for this measurement (e.g. pre-elab / post-elab)")
    a = ap.parse_args()

    cat = load_catalog(Path(a.catalog))
    rows = []
    for t in cat.templates:
        legacy = (t.verbal_template or "").strip()
        frames = [f for f in (t.frames() or []) if f and f.strip() != legacy]
        if not legacy or not frames:
            continue
        rows.append({"template": t.template_id, "legacy": legacy,
                     "frames": frames, "manchester": t.manchester_template or ""})
    print(f"{len(rows)} templates with legacy + frames", flush=True)

    # 1. perplexity — every template, frames vs legacy
    print("scoring perplexity (AEGIR_PPL_MODEL, default Qwen3-1.7B-Base @ cuda:4)…", flush=True)
    leg_ppl = Q.perplexity([Q.fill_slots(r["legacy"]) for r in rows])
    frm_ppl = []
    for r in rows:
        ps = Q.perplexity([Q.fill_slots(f) for f in r["frames"]])
        frm_ppl.append(min(ps))                       # a template is as natural as its BEST frame
    pairs = [(l, f) for l, f in zip(leg_ppl, frm_ppl) if l == l and f == f]  # drop NaN
    ratio = st.median(f / l for l, f in pairs)
    ppl = {"n": len(pairs), "legacy_median": round(st.median(l for l, _ in pairs), 1),
           "frames_best_median": round(st.median(f for _, f in pairs), 1),
           "frames_over_legacy_median_ratio": round(ratio, 3)}
    print(f"  ppl legacy median {ppl['legacy_median']} · frames(best) median {ppl['frames_best_median']} "
          f"· ratio {ppl['frames_over_legacy_median_ratio']}", flush=True)

    # 2. LLM-as-judge on a seeded sample (candidate = seeded frame; baseline = legacy)
    rng = random.Random(a.seed)
    jrows = []
    if a.judge_sample:
        for r in rng.sample(rows, min(a.judge_sample, len(rows))):
            cand = rng.choice(r["frames"])
            v = Q.judge(r["manchester"], cand, r["legacy"], cand_is_a=rng.random() < 0.5)
            if v:
                jrows.append({**v, "template": r["template"], "candidate": cand, "baseline": r["legacy"],
                              "manchester": r["manchester"]})
            print(f"  judged {len(jrows)}", end="\r", flush=True)
        (REPO / "build/verbalization_judgments.jsonl").write_text(
            "\n".join(json.dumps(j) for j in jrows))
    judge = None
    if jrows:
        wins = sum(1 for j in jrows if j["prefer"] == "candidate")
        ties = sum(1 for j in jrows if j["prefer"] == "tie")
        scored = [j for j in jrows if j.get("faithful") is not None]
        judge = {"n": len(jrows), "n_scored": len(scored),
                 "faithful_mean": round(st.mean(j["faithful"] for j in scored), 2) if scored else None,
                 "natural_mean": round(st.mean(j["natural"] for j in scored), 2) if scored else None,
                 "winrate_vs_legacy": round((wins + 0.5 * ties) / len(jrows), 3)}
        print(f"\n  judge n={judge['n']} faithful {judge['faithful_mean']} natural {judge['natural_mean']} "
              f"winrate {judge['winrate_vs_legacy']}", flush=True)

    # 3. blinded human sheet
    sheet = None
    if jrows and a.human_sample:
        sheet = str(Q.human_sample(jrows, REPO / "build/verbalization_human_sample.md",
                                   k=a.human_sample, seed=a.seed))
        print(f"  human sheet → {sheet} (+ .key.json)", flush=True)

    out = Path(a.out)
    out.write_text(json.dumps({"tag": a.tag, "catalog": a.catalog, "ppl": ppl, "judge": judge,
                               "human_sheet": sheet, "seed": a.seed}, indent=1))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
