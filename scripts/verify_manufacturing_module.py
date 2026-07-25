#!/usr/bin/env python
"""verify_manufacturing_module — increment (b)'s isolated gate (#44).

Assembles an ISOLATED ontology (never production): the numeric BFO backbone +
the manufacturing module → HermiT consistency (budget-guarded) → kvasir check +
ddl → the FIRST PARITY DIFF against RH's DDL yardstick (work_orders is the one
table present in both; the logistics tables extend the yardstick per its own
noted gap). Artifacts: build/manufacturing_module_verify.json + build/
manufacturing_module.omn + build/manufacturing_module_ddl.sql.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run python scripts/verify_manufacturing_module.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.elements_backbone import ELEMENTS_BACKBONE_OMN  # noqa: E402
from aegir.ontology.manufacturing_module import MANUFACTURING_OMN  # noqa: E402

_GATED_OMN = MANUFACTURING_OMN + ELEMENTS_BACKBONE_OMN  # the fail-scope: OUR modules

_PREFIXES = ("Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
             "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
             "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
             "Prefix: skos: <http://www.w3.org/2004/02/skos/core#>\n"
             "Prefix: sdg: <https://signals.zndx.org/sdg#>\n"
             "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
             "Prefix: cco: <https://www.commoncoreontologies.org/>\n"
             "Ontology: <https://signals.zndx.org/sdg/manufacturing-module-verify>\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    bro = _load("bro", REPO / "scripts/build_realized_ontology.py")
    et = _load("et", REPO / "scripts/emit_taxonomy.py")
    # NUMERIC_BFO already carries the module (unconditional critical-path append) —
    # never append it twice; referenced externals absent from the backbone are
    # declared WITH their authoritative labels (any-valid-OMN doctrine: the
    # ontology carries the knowledge; kvasir's label machinery does the rest)
    from aegir.ontology.external_index import declare_externals
    extra = declare_externals(c for c in ("cco:ont00000192", "cco:ont00000853",
                                          "cco:ont00001359")
                              if f"Class: {c}" not in bro.NUMERIC_BFO)
    omn = _PREFIXES + bro.NUMERIC_BFO + "\n" + extra
    omn_p = REPO / "build/manufacturing_module.omn"
    omn_p.write_text(omn)

    v = et.hermit(omn, budget_s=600)
    print(f"HermiT: consistent={v.get('consistent')} · unsat={len(v.get('unsat', []))} "
          f"{v.get('unsat', [])[:6]}")

    kv = REPO / "components/kvasir/target/release/kvasir"
    chk = subprocess.run([str(kv), "lower", str(omn_p), "--json"],
                         capture_output=True, text=True)
    print(f"kvasir lower: exit {chk.returncode}"
          + (f" · {chk.stderr.strip()[:160]}" if chk.returncode else ""))
    plan_p = REPO / "build/manufacturing_module_plan.json"
    ddl = subprocess.run([str(kv), "ddl", str(omn_p), "--sql", "--plan", str(plan_p)],
                         capture_output=True, text=True)
    ddl_sql = ddl.stdout if ddl.returncode == 0 else ""
    (REPO / "build/manufacturing_module_ddl.sql").write_text(ddl_sql)
    plan = json.loads(plan_p.read_text()) if plan_p.exists() else {}
    tables = [t.get("name") for t in plan.get("tables", [])]
    print(f"kvasir ddl: exit {ddl.returncode} · {len(tables)} tables: {tables}")

    # UNDERSPECIFICATION (RH 2026-07-25, native — no aegir-side shim): kvasir's
    # Plan.underspecified IS the signal, co-equal with unsat. This gate FAILS
    # only on THIS MODULE's classes (scoping which signals block is the gate's
    # job; producing signals is kvasir's alone). Everything else sdg-scoped is
    # REPORTED as the standing worklist — annotation-layer vocab (relational
    # concepts, spec mappings) is increment (c)'s subject: reference-table rows,
    # not entity tables.
    import re as _re
    module_frags = set(_re.findall(r"Class:\s+sdg:(\w+)", _GATED_OMN))
    sdg_under = sorted(c.rsplit("#", 1)[-1] for c in plan.get("underspecified", [])
                       if "signals.zndx.org/sdg#" in c)
    gate_under = sorted(set(sdg_under) & module_frags)
    worklist = sorted(set(sdg_under) - module_frags)
    if gate_under:
        print(f"⚑ UNDERSPECIFIED module classes (kvasir-native — remediate): {gate_under}")
    else:
        print(f"underspecification (kvasir-native): module NONE — all {len(module_frags)} "
              f"classes materialize · assembly worklist {len(worklist)} "
              f"(annotation-layer vocab → increment (c) reference tables)")

    (REPO / "build/manufacturing_module_verify.json").write_text(json.dumps(
        {"hermit": {k: v.get(k) for k in ("consistent", "exit", "unsat")},
         "kvasir_lower_exit": chk.returncode, "kvasir_ddl_exit": ddl.returncode,
         "tables": tables, "underspecified_module": gate_under,
         "underspecified_worklist": worklist,
         "underspecified_total": len(plan.get("underspecified", []))}, indent=1))
    print("→ build/manufacturing_module_verify.json")
    return 0 if (v.get("consistent") and not v.get("unsat") and ddl.returncode == 0
                 and not gate_under) else 1


if __name__ == "__main__":
    raise SystemExit(main())
