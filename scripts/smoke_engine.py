#!/usr/bin/env python
"""End-to-end smoke test for the closed generate→re-ground→refine engine.

Deterministic (no LLM, no BERTopic): injects a controlled ``generate_fn`` and
``topic_recovery_fn`` to drive 10 episodes through the full skill library + engine,
each engineered to a known outcome, and asserts the hard-gate contract:

  (a) a chapter is admitted ONLY when every unit passes its per-modality hard gates
      AND topic_recovery ≥ τ_topic (the hard conjunction; F is never consulted);
  (b) admitted chapters satisfy topic_recovery ≥ 0.80 AND r_axiom ≥ 0.45;
  (c) the per-unit repair loop converges (a fixable unit → admitted after repair);
  (d) no regression vs the S2 baseline (admitted chapters' table r_axiom == 1.000).

Scenarios: 6 clean-pass, 1 repair-then-pass (prose ungrounded then grounded),
3 designed failures (fail_axiom / fail_ground / fail_topic).

Usage::  uv run --no-sync python scripts/smoke_engine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.engine import Seed, run_episode  # noqa: E402
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
from aegir.ontology.skills.base import Claim, Gates  # noqa: E402
from aegir.ontology.skills.s2_relational_table import SourceSpan  # noqa: E402
from aegir.ontology.type_check import ColumnSpec, FKEdge, TableSpec, UnitSchema  # noqa: E402

CLASS_VALS = ["Mammal", "Organism", "Process"]
DATA_VALS = ["42", "2021-01-01", "7.5"]


def _is_data(ot: str) -> bool:
    return ot not in ("Class", "Individual", "NamedIndividual", "ObjectProperty")


def correct_schema(t: CatalogTemplate) -> UnitSchema:
    tn = f"t_{t.template_id}"
    col_slots = [(s, ot) for s, ot in t.slot_types.items() if ot != "ObjectProperty"]
    op_slots = [s for s, ot in t.slot_types.items() if ot == "ObjectProperty"]
    cols = [ColumnSpec(name=s, slot_type=ot, slot_ref=s) for s, ot in col_slots]
    rows = [[(DATA_VALS[r % 3] if _is_data(ot) else CLASS_VALS[(r + i) % 3])
             for i, (s, ot) in enumerate(col_slots)] for r in range(3)]
    fks = []
    for s in op_slots:
        if len(cols) >= 2:
            fks.append(FKEdge(tn, cols[0].name, tn, cols[1].name, s))
        elif cols:
            fks.append(FKEdge(tn, cols[0].name, tn, cols[0].name, s))
    return UnitSchema([TableSpec(tn, t.template_id, cols, rows)], fks)


def mistyped_schema(t: CatalogTemplate) -> UnitSchema:
    tn = f"t_{t.template_id}"
    col_slots = [(s, ot) for s, ot in t.slot_types.items() if ot != "ObjectProperty"]
    cols = [ColumnSpec(name=s, slot_type=ot, slot_ref=s) for s, ot in col_slots]
    cols.append(ColumnSpec("junk", "Class", "__nonexistent__"))
    rows = [["9", "9", "9", "9", "9"][: len(cols)] for _ in range(3)]
    return UnitSchema([TableSpec(tn, t.template_id, cols, rows)], [])


def correct_edges(t: CatalogTemplate) -> list[tuple[str, str]]:
    cs = [s for s, ot in t.slot_types.items() if ot != "ObjectProperty"]
    return [(cs[0], cs[1])] if len(cs) >= 2 else []


def make_gen(scenario: str, evidence: list[SourceSpan], tmpl: CatalogTemplate):
    ev0 = evidence[0].text
    state = {"s1": 0}

    def gen(skill_id, refs, ev, repair=False):
        if skill_id == "S2":
            return mistyped_schema(tmpl) if scenario == "fail_axiom" else correct_schema(tmpl)
        if skill_id in ("S1", "S4", "S6"):
            if scenario == "fail_ground":
                return ("ungrounded prose", [Claim("an assertion with no evidence", None)])
            if scenario == "repair" and skill_id == "S1":
                state["s1"] += 1
                if state["s1"] == 1 and not repair:
                    return ("ungrounded first try", [Claim("xyzzy ungrounded", None)])
                return (f"grounded: {ev0}", [Claim(ev0, None)])
            return (f"prose grounded in {ev0}", [Claim(ev0, None)])
        if skill_id == "S3":
            edges = [("bogusA", "bogusB")] if scenario == "fail_cross" else correct_edges(tmpl)
            return ("graph TD", edges, [])
        return ("", [])

    return gen


def make_tr(scenario: str):
    return lambda chapter_md, target: 0.30 if scenario == "fail_topic" else 0.90


def pick_template(catalog) -> CatalogTemplate:
    op = [t for t in catalog.templates
          if "ObjectProperty" in t.slot_types.values()
          and sum(1 for v in t.slot_types.values() if v != "ObjectProperty") >= 2]
    return op[0] if op else next(t for t in catalog.templates if len(t.slot_types) >= 2)


def main() -> int:
    catalog = load_catalog(REPO / "src/aegir/ontology/catalog/combined.json")
    tmpl = pick_template(catalog)
    evidence = [SourceSpan("ev0", "photosynthesis converts light energy"),
                SourceSpan("ev1", "cellular respiration releases energy")]
    print(f"template: {tmpl.template_id}  slot_types={tmpl.slot_types}\n")

    scenarios = ["pass"] * 6 + ["repair", "fail_axiom", "fail_ground", "fail_topic"]
    expect = {"pass": True, "repair": True,
              "fail_axiom": False, "fail_ground": False, "fail_topic": False}

    print(f"{'#':>2} {'scenario':12} {'admit':>5} {'exp':>5} {'tr':>5} {'r_ax':>6}  match")
    results = []
    for i, sc in enumerate(scenarios):
        seed = Seed(f"cluster{i}", "topic_photosynthesis", [tmpl.template_id], evidence)
        res = run_episode(seed, catalog, make_gen(sc, evidence, tmpl), make_tr(sc),
                          Gates(), max_repair=2)
        match = res.admitted == expect[sc]
        results.append((sc, res, match))
        print(f"{i:>2} {sc:12} {str(res.admitted):>5} {str(expect[sc]):>5} "
              f"{res.topic_recovery:>5.2f} {res.r_axiom:>6.3f}  {'ok' if match else 'XX'}")

    # ── assertions ──
    admitted = [(sc, r) for sc, r, _ in results if r.admitted]
    repair_res = next(r for sc, r, _ in results if sc == "repair")
    checks = {
        "admission matches expectation (gate logic)": all(m for _, _, m in results),
        "admitted ⇒ topic_recovery ≥ 0.80": all(r.topic_recovery >= 0.80 for _, r in admitted),
        "admitted ⇒ r_axiom ≥ 0.45": all(r.r_axiom >= 0.45 for _, r in admitted),
        "no regression vs S2 baseline (admitted r_axiom == 1.000)":
            all(abs(r.r_axiom - 1.0) < 1e-6 for _, r in admitted),
        "repair loop exercised + converged":
            repair_res.admitted and any("repair)" in d and "(0 repair" not in d
                                        for d in repair_res.decisions),
    }
    print("\n=== checks ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print("\nrepair-seed decisions:")
    for d in repair_res.decisions:
        print(f"    {d}")
    print("\nSMOKE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
