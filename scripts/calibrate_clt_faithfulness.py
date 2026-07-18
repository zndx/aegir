#!/usr/bin/env python
"""calibrate_clt_faithfulness — set the CLT feature-overlap floor from the old-prompt contrast set (#29).

The stopped batch's pre-membrane elaborations (build/verbalization_elaborations.old-prompt.json) are a
ready-made calibration corpus: for each (base, elaboration) pair we compute the CLT feature-overlap
J(base, elab) on cuda:4, label a judged sample with the engine judge (silver labels: faithful ≥4 → pos,
≤2 → neg), and measure the UNRELATED floor from cross-template base pairs. τ_clt is chosen to maximize
balanced accuracy on the silver labels. Output → build/clt_faithfulness_calibration.json, consumed by
the membrane-inline elaboration loop (elaborate v2): CLT floor first (cheap, judge-free), judge only on
the margin band.

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/calibrate_clt_faithfulness.py --sample 60
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", default=str(REPO / "build/verbalization_elaborations.old-prompt.json"))
    ap.add_argument("--sample", type=int, default=60, help="elaborations to judge (silver labels)")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    import torch
    from circuit_tracer import ReplacementModel
    clt = ReplacementModel.from_pretrained("Qwen/Qwen3-1.7B-Base",
                                           "bluelightai/clt-qwen3-1.7b-base-20k", dtype=torch.bfloat16)
    _cache: dict = {}

    def feats(text: str):
        if text not in _cache:
            toks = clt.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=192).input_ids.to(clt.cfg.device)
            f = clt.get_activations(toks)[1]                  # [layers, pos, 20480]
            _cache[text] = set(map(tuple, (f != 0).any(dim=1).nonzero().tolist()))
        return _cache[text]

    def J(x: str, y: str) -> float:
        fx, fy = feats(x), feats(y)
        return len(fx & fy) / max(1, len(fx | fy))

    d = json.loads(Path(a.contrast).read_text())
    rng = random.Random(a.seed)
    entries = [(k, v) for k, v in d.items() if v.get("elaborations")]
    pairs = [(k, v["base"], e) for k, v in entries for e in v["elaborations"]]
    print(f"{len(entries)} templates · {len(pairs)} (base, elaboration) pairs", flush=True)

    # 1. unrelated floor — cross-template base pairs
    bases = [v["base"] for _, v in rng.sample(entries, min(40, len(entries)))]
    unrel = [J(bases[i], bases[(i + 7) % len(bases)]) for i in range(len(bases))]
    print(f"unrelated floor: median {st.median(unrel):.3f} · p90 {sorted(unrel)[int(.9 * len(unrel))]:.3f}",
          flush=True)

    # 2. silver labels via the engine judge on a sample
    from aegir.ontology.verbalization_quality import judge
    sample = rng.sample(pairs, min(a.sample, len(pairs)))
    rows = []
    for k, base, elab in sample:
        ov = J(base, elab)
        v = judge(base, elab, base, cand_is_a=rng.random() < 0.5)
        faithful = v.get("faithful") if v else None
        rows.append({"key": k, "overlap": round(ov, 4), "faithful": faithful,
                     "elab": elab[:140]})
        print(f"  scored {len(rows)} (J={ov:.3f} faithful={faithful})", end="\r", flush=True)
    pos = [r["overlap"] for r in rows if (r["faithful"] or 0) >= 4]
    neg = [r["overlap"] for r in rows if r["faithful"] is not None and r["faithful"] <= 2]
    print(f"\nsilver: {len(pos)} faithful · {len(neg)} unfaithful · "
          f"{len(rows) - len(pos) - len(neg)} excluded/unscored", flush=True)

    # 3. τ by balanced accuracy over candidate thresholds
    best = None
    for t in [round(0.02 * i, 2) for i in range(1, 40)]:
        if not pos or not neg:
            break
        tpr = sum(1 for x in pos if x >= t) / len(pos)
        tnr = sum(1 for x in neg if x < t) / len(neg)
        ba = (tpr + tnr) / 2
        if best is None or ba > best[1]:
            best = (t, ba, tpr, tnr)
    out = {"n_pairs": len(pairs), "judged": len(rows),
           "unrelated_floor": {"median": round(st.median(unrel), 4)},
           "pos": {"n": len(pos), "median": round(st.median(pos), 4) if pos else None},
           "neg": {"n": len(neg), "median": round(st.median(neg), 4) if neg else None},
           "tau_clt": best[0] if best else None,
           "balanced_accuracy": round(best[1], 3) if best else None,
           "tpr": round(best[2], 3) if best else None, "tnr": round(best[3], 3) if best else None,
           "rows": rows}
    (REPO / "build/clt_faithfulness_calibration.json").write_text(json.dumps(out, indent=1))
    if best:
        print(f"τ_clt = {best[0]} (balanced acc {best[1]:.3f} · TPR {best[2]:.2f} · TNR {best[3]:.2f})")
    print("→ build/clt_faithfulness_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
