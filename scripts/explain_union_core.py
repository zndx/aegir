#!/usr/bin/env python
"""explain_union_core — HermiT justifications for the union's taxonomy core.

Builds the taxonomy-core document from the current union (probe_union_tbox machinery),
reasons it (fast — no restrictions), and emits minimal justifications for every unsat
core class → build/taxcore_justifications.json, the input of derive_category_cuts.py.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run python scripts/explain_union_core.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_realize/sdg-ontology.omn")
    ap.add_argument("--out", type=Path, default=REPO / "build/taxcore_justifications.json")
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--full", action="store_true",
                    help="explain on the FULL TBox (restriction-mediated defects — "
                         "functional-property intersections, domain/range forcing — are "
                         "invisible in the taxonomy core); targets spread over the unsat list")
    a = ap.parse_args()

    from aegir.ontology.deeponto_harness import ensure_jvm
    ensure_jvm()
    spec = importlib.util.spec_from_file_location("bro", REPO / "scripts/build_realized_ontology.py")
    assert spec and spec.loader
    bro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bro)
    from probe_union_tbox import strip_abox, taxonomy_core

    doc = strip_abox(a.omn.read_text())
    if not a.full:
        doc = taxonomy_core(doc)
    onto, _path, consistent, n, unsat, _ = bro._reason(doc)
    print(f"core: consistent={consistent} n={n} unsat={len(unsat)}", flush=True)
    why = {}
    if unsat:
        from aegir.ontology.explain import explain_unsatisfiable
        targets = unsat
        if a.full and len(unsat) > a.cap:
            step = max(1, len(unsat) // a.cap)
            targets = unsat[::step][:a.cap]      # spread, not head — diverse defect coverage
        why = explain_unsatisfiable(onto.owl_onto, onto.reasoner.owl_reasoner, targets, cap=a.cap)
    a.out.write_text(json.dumps({"unsat_n": len(unsat), "why": why}, indent=1))
    print(f"explained {len(why)} → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
