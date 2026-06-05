#!/usr/bin/env python
"""Smoke test for S2 (synth-relational-table-with-cross-FKs) + the r_axiom type-checker.

Deterministic (no LLM): validates the fidelity-contract machinery the live run will
rely on. It confirms —
  (a) an explicit, type-correct S2 unit scores materially higher r_axiom than the
      prior prose-only baseline (ablation_v1 verification.parquet r_axiom), and a
      mistyped unit scores low; and
  (b) the per-unit repair loop converges (mistyped → regenerate → admitted).

What this proves: the type-checker rewards correct slot structure, the explicit
column→slot_ref schema is the source of the lift, and the repair loop behaves. What
it does NOT prove: that the LLM backend produces type-correct tables — that is the
live run, once default_generate_fn is bound to the GLM/Grok mix.

Usage::  uv run --no-sync python scripts/smoke_s2.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
from aegir.ontology.skills.s2_relational_table import (  # noqa: E402
    SourceSpan,
    synth_relational_table,
)
from aegir.ontology.type_check import (  # noqa: E402
    ColumnSpec,
    FKEdge,
    TableSpec,
    UnitSchema,
    r_axiom,
)

CLASS_VALS = ["Mammal", "Organism", "Process", "InformationEntity", "Quality", "Disposition"]
DATA_VALS = ["42", "2021-03-14", "7.5", "100", "2019-01-01"]


def _is_data(owl_type: str) -> bool:
    return owl_type not in ("Class", "Individual", "NamedIndividual", "ObjectProperty")


def build_correct(t: CatalogTemplate) -> UnitSchema:
    """An ideal, type-correct S2 unit for template t (the explicit-schema ceiling)."""
    tname = f"t_{t.template_id}"
    col_slots = [(s, ot) for s, ot in t.slot_types.items() if ot != "ObjectProperty"]
    op_slots = [s for s, ot in t.slot_types.items() if ot == "ObjectProperty"]
    cols = [ColumnSpec(name=s, slot_type=ot, slot_ref=s) for s, ot in col_slots]
    rows = []
    for r in range(3):
        row = [(DATA_VALS[r % len(DATA_VALS)] if _is_data(ot)
                else CLASS_VALS[(r + i) % len(CLASS_VALS)])
               for i, (s, ot) in enumerate(col_slots)]
        rows.append(row)
    fks = []
    for s in op_slots:
        if len(cols) >= 2:
            fks.append(FKEdge(tname, cols[0].name, tname, cols[1].name, s))
        elif cols:
            fks.append(FKEdge(tname, cols[0].name, tname, cols[0].name, s))
    return UnitSchema([TableSpec(tname, t.template_id, cols, rows)], fks)


def build_mistyped(t: CatalogTemplate) -> UnitSchema:
    """Class columns filled with numbers (type mismatch), no FKs (object-properties
    unfulfilled), plus an ungrounded column — should score near zero."""
    tname = f"t_{t.template_id}"
    col_slots = [(s, ot) for s, ot in t.slot_types.items() if ot != "ObjectProperty"]
    cols = [ColumnSpec(name=s, slot_type=ot, slot_ref=s) for s, ot in col_slots]
    cols.append(ColumnSpec(name="junk", slot_type="Class", slot_ref="__nonexistent_slot__"))
    rows = [["999", "123", "7", "55", "8"][: len(cols)] for _ in range(3)]
    return UnitSchema([TableSpec(tname, t.template_id, cols, rows)], [])  # no FKs


def pick_template(catalog) -> CatalogTemplate:
    """Prefer a template with an ObjectProperty slot + ≥2 Class slots (exercises FK
    and column paths); fall back to ≥2 slots."""
    has_op = [t for t in catalog.templates
              if "ObjectProperty" in t.slot_types.values()
              and sum(1 for v in t.slot_types.values() if v != "ObjectProperty") >= 2]
    if has_op:
        return has_op[0]
    multi = [t for t in catalog.templates if len(t.slot_types) >= 2]
    return multi[0] if multi else catalog.templates[0]


def main() -> int:
    catalog = load_catalog(REPO / "src/aegir/ontology/catalog/combined.json")
    tmpl = pick_template(catalog)
    print(f"template: {tmpl.template_id}  slot_types={tmpl.slot_types}")

    correct = build_correct(tmpl)
    mistyped = build_mistyped(tmpl)
    r_correct, bd_c = r_axiom(correct, catalog)
    r_mistyped, bd_m = r_axiom(mistyped, catalog)

    # Prior prose-only baseline: ablation_v1 full-arm r_axiom (verification.parquet).
    baseline = None
    vp = glob.glob("/raid/checkpoints/aegir-artifacts/ablation_v1/6e6901e291ef1f87/verification.parquet")
    if vp:
        import pyarrow.parquet as pq
        baseline = float(pq.read_table(vp[0]).to_pandas()["r_axiom"].mean())

    print(f"\nr_axiom  type-correct S2 unit : {r_correct:.3f}")
    print(f"r_axiom  mistyped unit         : {r_mistyped:.3f}")
    print(f"r_axiom  prose-only baseline   : "
          f"{baseline:.3f} (ablation_v1 full arm)" if baseline is not None
          else "r_axiom  prose-only baseline   : (verification.parquet not found; expected ~0.50)")
    for c in bd_c.offending():
        print(f"   correct-unit miss: {c.table}/{c.slot} — {c.reason}")

    # Repair loop: mistyped first, type-correct on repair.
    state = {"n": 0}

    def gen(prompt, refs, evidence, repair=None):
        state["n"] += 1
        return build_mistyped(tmpl) if state["n"] == 1 else build_correct(tmpl)

    res = synth_relational_table([tmpl.template_id], [SourceSpan("doc0", "evidence")],
                                 catalog, generate_fn=gen, tau_axiom=0.45, max_repair=2)
    print(f"\nrepair loop: history={[round(h, 3) for h in res.history]}  "
          f"attempts={res.attempts}  admitted={res.admitted}  final={res.r_axiom:.3f}")

    # ── assertions ──
    base = baseline if baseline is not None else 0.50
    checks = {
        "type-correct unit high r_axiom (≥0.90)": r_correct >= 0.90,
        "type-correct materially > baseline (Δ≥0.30)": r_correct - base >= 0.30,
        "mistyped unit low r_axiom (<0.45)": r_mistyped < 0.45,
        "repair converged to admitted": res.admitted and res.attempts >= 1,
        "repair improved r_axiom": res.history[-1] > res.history[0],
    }
    print("\n=== checks ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print("\nSMOKE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
