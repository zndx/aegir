#!/usr/bin/env python
"""mediate — run the agent-mediated meta-harness on gap topics (inc-1).

Wires the RETE/FSM spine (src/aegir/meta_harness/fsm_rete) to real effectors:
mint = ACP/Grok (scripts/mediate_acp), gate = ContractGate (scripts/mediate_gate).
The inaugural capability proof: does agent-mediation drive register-fair coverage
(R1) to CI-clean-positive on the gap topics the one-shot generator failed (on-topic
0.007, Δ+0.003, CI touching 0)? Logs trace + scorecard; stages admitted constructs
to a .candidate catalog (charter: review-before-promote). One grok session per topic
(within-topic refine continuity; fresh context across topics). The ContractGate is
built once (JVM + Vt + seed pool) and shared.

Run (needs the JVM libs for DeepOnto):
  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs:$(pwd)/build/cuda-driver-libs \
    uv run --no-sync python scripts/mediate.py \
    --coverage-run /raid/.../coverage_v1/043d7dcc185245c8 --topics 124 --max-iters 6
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from aegir.meta_harness.fsm_rete import MetaHarness, seed_rules  # noqa: E402
import coverage_r1 as cr  # noqa: E402
from mediate_gate import ContractGate  # noqa: E402
from mediate_acp import ACPMintEffector  # noqa: E402

_FAIL_SIG = {"has_construct": True, "deeponto_ok": False, "deeponto_complex": False,
             "polyglot_ok": False, "novelty_ok": False, "schema_ok": False,
             "r1_on": 0.0, "r1_ci_low": -1.0, "n_cols": 0}


class MediateEffector:
    """Adapts the spine's single-Effector protocol to the two real components."""

    def __init__(self, mint_eff, gate_eff):
        self.m, self.g = mint_eff, gate_eff

    def mint(self, topic, objective, feedback, prev):
        return self.m.mint(topic, objective, feedback, prev)

    def gate(self, construct, topic):
        if not construct or construct.get("template") is None:
            return dict(_FAIL_SIG)          # mint failed → drive a retry, bounded by max_iters
        return self.g.gate(construct, topic)


def batch_r1(admitted: list[dict], Vt, rng) -> dict:
    """The actual G-cov verdict: pooled on-vs-shuffled R1 across admitted constructs."""
    deltas = []
    for a in admitted:
        t, tid = a["template"], int(a["topic_id"])
        vc = cr.construct_terms(t)
        if not vc or tid >= len(Vt):
            continue
        on = cr.r1(vc, Vt[tid], use_embed=False)
        others = [j for j in range(len(Vt)) if j != tid and Vt[j]]
        sh = np.mean([cr.r1(vc, Vt[j], use_embed=False)
                      for j in rng.choice(others, size=min(40, len(others)), replace=False)])
        deltas.append(on - float(sh))
    if not deltas:
        return {"n": 0}
    m, lo, hi = cr.boot_ci(np.array(deltas), rng)
    return {"n": len(deltas), "delta": m, "ci": [lo, hi], "ci_clean": lo > 0,
            "on_mean": float(np.mean([d for d in deltas]))}


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-run", required=True)
    ap.add_argument("--topics", required=True, help="comma-separated topic_ids")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260616)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    topic_ids = [int(x) for x in args.topics.split(",")]
    out = Path(args.out or f"/raid/checkpoints/aegir-artifacts/evidence/meta_harness/run_{topic_ids[0]}")
    out.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    rows = pq.read_table(Path(args.coverage_run) / "topic_coverage.parquet").to_pylist()
    by = {r["topic_id"]: r for r in rows}
    reprs = [(by.get(i, {}).get("topic_repr_text") or "") for i in range(len(rows))]
    Vt = cr.topic_term_signatures(reprs)

    gate = ContractGate(args.coverage_run)           # build once (JVM + Vt + seed pool)
    scorecard, admitted = [], []
    for tid in topic_ids:
        topic = dict(by[tid]); topic["_top_terms"] = sorted(Vt[tid])
        eff = ACPMintEffector()
        spine = MetaHarness(seed_rules(), MediateEffector(eff, gate), max_iters=args.max_iters,
                            trace_path=str(out / f"trace_t{tid}.jsonl"))
        try:
            ctx = spine.run(topic)
        finally:
            eff.close()
        sig = spine.wm.signals()
        c = ctx.construct or {}
        row = {"topic_id": tid, "outcome": ctx.terminate, "iters": ctx.iterations,
               "construct_id": c.get("template_id"),
               "r1_on": sig.get("r1_on"), "r1_ci_low": sig.get("r1_ci_low"),
               "deeponto_ok": sig.get("deeponto_ok"), "deeponto_complex": sig.get("deeponto_complex"),
               "polyglot_ok": sig.get("polyglot_ok"), "novelty_ok": sig.get("novelty_ok"),
               "schema_ok": sig.get("schema_ok"), "n_cols": sig.get("n_cols"),
               "manchester": (c.get("template").manchester_template if c.get("template") else None)}
        scorecard.append(row)
        print(f"\n[t{tid}] {ctx.terminate} in {ctx.iterations} iters | r1_on={row['r1_on']} "
              f"ci_low={row['r1_ci_low']} | {row['construct_id']}")
        if ctx.terminate == "promote" and c.get("template") is not None:
            admitted.append({"template": c["template"], "topic_id": tid})

    verdict = batch_r1(admitted, Vt, rng)
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    (out / "candidates.candidate.json").write_text(json.dumps(
        {"version": "0.1.0-mediate-candidate",
         "templates": [asdict(a["template"]) for a in admitted], "null_stats": {}}, indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"\n=== MEDIATE VERDICT ({len(admitted)}/{len(topic_ids)} promoted) ===")
    if verdict.get("n"):
        print(f"  batch R1 (admitted): on_mean={verdict['on_mean']:.3f}  "
              f"Δ(on−shuffled)={verdict['delta']:+.3f} [{verdict['ci'][0]:+.3f},{verdict['ci'][1]:+.3f}]  "
              f"{'CI-CLEAN ✓' if verdict['ci_clean'] else 'spans 0'}")
        print(f"  vs one-shot generator: Δ+0.003 (CI touched 0). "
              f"{'agent-mediation CLEARS R1 where one-shot failed.' if verdict['ci_clean'] else 'not yet CI-clean.'}")
    else:
        print("  no constructs promoted — see per-topic scorecard for where the contract blocked.")
    print(f"  artifacts → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
