#!/usr/bin/env python
"""mediate_h0 — run the H₀ harness over gap topics via the candidate filesystem (inc-2b).

The Meta-Harness-FORM driver. Where inc-1 (mediate.py) ran the FSM spine, this stands
up the candidate filesystem (candidates/000/ = the committed H₀ seed), validates the
interface with fixtures (cheap, before any spend), then EVALUATES H₀ over the search-set
topics wiring the two FROZEN executors:
    complete = ACPMintEffector.complete   (raw Grok completion over ACP)
    gate     = ContractGate.gate          (the committed verifiers + HermiT coherence)
It writes per-topic traces + scores.json (reward = batch on-vs-shuffled R1; cost = mint
calls) — the uncompressed feedback channel the inc-2c proposer will navigate. Acid test:
H₀ reproduces inc-1 (t124 → promote, r1≈0.39) from ONE readable harness file.

NOTE: one persistent grok session across the search set (fine for the single-topic acid
test; per-topic session freshness is an inc-2c refinement when the set is large).

Run (JVM libs for DeepOnto/HermiT):
  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs:$(pwd)/build/cuda-driver-libs \
    uv run --no-sync python scripts/mediate_h0.py \
    --coverage-run /raid/.../coverage_v1/043d7dcc185245c8 --topics 124 --max-iters 6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import coverage_r1 as cr  # noqa: E402
import harness_search as hs  # noqa: E402
from mediate_gate import ContractGate  # noqa: E402
from mediate_acp import ACPMintEffector  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-run", required=True)
    ap.add_argument("--topics", required=True, help="comma-separated topic_ids (the search set)")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--seed", type=int, default=20260616)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    topic_ids = [int(x) for x in args.topics.split(",")]
    ws = Path(args.workspace
              or f"/raid/checkpoints/aegir-artifacts/evidence/meta_harness/h0_{topic_ids[0]}")
    ws.mkdir(parents=True, exist_ok=True)

    # 1. candidate filesystem + interface validation (cheap, before any Grok/JVM spend)
    d0 = hs.init_search(ws)
    ok, msg = hs.validate_harness(d0)
    print(f"[validate] candidates/000 (H₀): {'PASS' if ok else 'FAIL'} — {msg}")
    if not ok:
        return 1

    # 2. topics + Vt from the coverage run
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(args.coverage_run) / "topic_coverage.parquet").to_pylist()
    by = {r["topic_id"]: r for r in rows}
    reprs = [(by.get(i, {}).get("topic_repr_text") or "") for i in range(len(rows))]
    Vt = cr.topic_term_signatures(reprs)
    topics = []
    for tid in topic_ids:
        t = dict(by[tid])
        t["_top_terms"] = sorted(Vt[tid])
        topics.append(t)

    # 3. FROZEN executors → evaluate H₀ over the search set
    gate = ContractGate(args.coverage_run)        # JVM + Vt + seed pool (build once)
    eff = ACPMintEffector()                         # one persistent grok session
    try:
        scores = hs.evaluate_harness(d0, topics, gate.gate, eff.complete, eff.exemplars,
                                     Vt, rng, max_iters=args.max_iters)
    finally:
        eff.close()

    # 4. report
    for pt in scores["per_topic"]:
        print(f"[t{pt['topic_id']}] {pt['outcome']} in {pt['iters']} iters | "
              f"r1_on={pt['r1_on']} ci_low={pt['r1_ci_low']} | {pt['construct_id']} | "
              f"objectives={pt['objectives']}")
    r, c = scores["reward"], scores["cost"]
    print(f"\n=== H₀ VERDICT ({c['promoted']}/{c['attempted']} promoted, "
          f"{c['mint_calls']} mint calls) ===")
    if r.get("n"):
        print(f"  batch R1 (promoted): on_mean={r['on_mean']:.3f}  Δ={r['delta']:+.3f} "
              f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  {'CI-CLEAN ✓' if r['ci_clean'] else 'spans 0'}")
        print("  (inc-1 FSM-spine reference: t124 promote, r1_on≈0.387, ci_low≈0.385)")
    else:
        print("  no constructs promoted — see per-topic scorecard / traces.")
    print(f"  artifacts → {d0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
