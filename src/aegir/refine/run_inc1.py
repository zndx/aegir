"""inc-1 — loop onto the LIVE corpus: real ontology tables + the taxonomy-bootstrapped value-ontology.

Run A: the membrane-gated loop on a clean `live_draft` (real `chapter_relational_payload` tables) with the
REAL `vibe-acp` agent at PROPOSE → the loop confirms the live tables are admissible (value/placeholder/RI gates
PASS on real ontology data) and the agent refines the prose. This is the loop running on live corpus structure.

Run B: a domain-contaminated live column (one cell sourced from a DIFFERENT admitted domain) → the VALUE_GATE
catches it via the *bootstrapped* value-ontology (inferred cross-domain disjointness) and the membrane
remediates. This proves the gate is live on the real 510-class fragment, not vacuous.

The remaining link to full live VALUE_GATE coverage is cell-source-from-mixing (retain which domain each
subtree-mixed value came from); flagged as the next sub-step. Run with `LD_LIBRARY_PATH=build/jvm-libs` + engine up.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from aegir.refine import eval as ev
from aegir.refine.live import live_draft
from aegir.refine.loop import deterministic_prose, run_refinement
from aegir.refine.value_ontology import DEFAULT_TAXONOMY

OUT = Path("/raid/build/aegir/path-a/refine")


def _report(tag: str, res: dict) -> None:
    b, r = res["baseline_metrics"], res["refined_metrics"]
    print(f"\n=== {tag} ===")
    print("fired:", " → ".join(res["fired"]), "| outcome:", res["outcome"], "| iters:", res["iterations"])
    print(f"  placeholder {b['placeholder_rate']}→{r['placeholder_rate']} | "
          f"disjointness {len(b['disjointness_violations'])}→{len(r['disjointness_violations'])} | "
          f"ri {b['ri_ok']}→{r['ri_ok']} | entailment {b['prose_entailment']}→{r['prose_entailment']} | "
          f"len {b['length']}→{r['length']} (ok {b['length_ok']}→{r['length_ok']})")


def _domain_contaminated(ch: dict) -> dict:
    """Retype one real column to an admitted-domain concept and source ONE cell from a different domain."""
    data = json.loads(Path(DEFAULT_TAXONOMY).read_text())["admitted"]
    mA = data[0]["members"][0]            # a leaf of hypernym A
    mB = data[1]["members"][0]            # a leaf of hypernym B (≠ A → disjoint)
    c = copy.deepcopy(ch)
    col = next(co for t in c["tables"] for co in t["columns"]
               if t.get("pk") != co["name"] and co["cells"])
    col["concept"] = mA
    for cell in col["cells"]:
        cell["source"] = mA
    col["cells"][0] = {"value": "cross_domain_value", "source": mB}    # the contaminant
    return c


def main() -> int:
    from aegir.refine.run_inc0d import agent_propose
    OUT.mkdir(parents=True, exist_ok=True)
    ch = live_draft(n_templates=2)
    print(f"live chapter: {ch['template_id']} | {len(ch['tables'])} tables, "
          f"{sum(len(c['cells']) for t in ch['tables'] for c in t['columns'])} cells | "
          f"baseline {ev.measure(ch)}")

    # Run A — loop onto the live corpus with the REAL agent
    resA = run_refinement(ch, propose_fn=agent_propose, max_iters=6,
                          trace_path=str(OUT / "inc1_live_agent.jsonl"), commit_dir=str(OUT))
    _report("RUN A — clean live draft, real vibe-acp PROPOSE", resA)
    print("  committed:", resA["committed"], "| live tables admissible (value/placeholder/RI gates passed):",
          resA["baseline_metrics"]["disjointness_violations"] == [] and resA["baseline_metrics"]["ri_ok"])

    # Run B — VALUE_GATE bites on a domain-contaminated live column (bootstrapped ontology)
    chB = _domain_contaminated(ch)
    resB = run_refinement(chB, propose_fn=deterministic_prose, max_iters=6,
                          trace_path=str(OUT / "inc1_live_contaminated.jsonl"))
    _report("RUN B — domain-contaminated live column, VALUE_GATE active", resB)
    b_caught = len(resB["baseline_metrics"]["disjointness_violations"]) > 0
    b_cleared = len(resB["refined_metrics"]["disjointness_violations"]) == 0
    print(f"  VALUE_GATE caught the cross-domain cell: {b_caught} "
          f"{resB['baseline_metrics']['disjointness_violations']}; remediated→0: {b_cleared}")

    ok = (resA["outcome"] == "promote" and b_caught and b_cleared
          and resB["outcome"] == "promote")
    print(f"\ninc-1 {'PASS' if ok else 'PARTIAL'}: loop ran on live ontology tables (A); "
          f"VALUE_GATE live on the bootstrapped fragment (B).")
    return 0 if ok else 1


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
