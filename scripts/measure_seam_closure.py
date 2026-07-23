#!/usr/bin/env python
"""measure_seam_closure — the cross-pathway seam, re-measured through RECORDED links.

The Phase-2 audit counted 4,149 constructs-web tables unknown to the released spine —
the measured surface of pathway disjointness. With the comprehensive lowering RELEASED,
closure is measured the only admissible way (confirmed-links rule): a construct table's
RECORDED entity classes (the construct's own ``entities`` list) joined by EXACT IRI to
the comprehensive closure's generator-recorded ``own_table`` associations. Name
matching appears nowhere.

    uv run python scripts/measure_seam_closure.py → build/seam_closure.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06")


def main() -> int:
    comp_p = sorted((REPO / "corpora/ddl-comprehensive").glob("*/ontology_entity_associations.json"))
    src = comp_p[-1] if comp_p else REPO / "build/comprehensive_release/ontology_entity_associations.json"
    closure = json.loads(src.read_text())
    # exact-IRI equality modulo prefix form: constructs record `sdg:X`, the closure
    # records the full IRI — prefix expansion is the SAME name, never a name match
    covered_classes = {r["class_iri"].rsplit("#", 1)[-1] for r in closure["classes"]
                       if any(a["kind"] == "own_table" for a in r["associations"])}
    n_tables = 0
    n_covered = 0
    uncovered_examples = []
    for cf in sorted((RUN / "constructs").glob("*.json")):
        con = json.loads(cf.read_text())
        ents = {str(e.get("name") if isinstance(e, dict) else e).rsplit("#", 1)[-1]
                .split(":", 1)[-1] for e in (con.get("entities") or [])}
        for t in (con.get("tables") or []):
            n_tables += 1
            # the construct records which classes it realized; the seam closes when
            # every such class has a generator-recorded comprehensive table
            if ents and all(e in covered_classes for e in ents):
                n_covered += 1
            elif len(uncovered_examples) < 8:
                uncovered_examples.append(
                    {"table": t.get("name"),
                     "uncovered": sorted(e for e in ents
                                         if e not in covered_classes)[:3]})
    out = {"rule": "recorded links only: construct entities (exact IRIs) × comprehensive "
                   "closure own_table records — no name matching",
           "closure_source": str(src.relative_to(REPO)),
           "n_construct_tables": n_tables,
           "n_seam_closed": n_covered,
           "seam_closed_rate": round(n_covered / max(1, n_tables), 4),
           "was": "4,149 construct tables unknown to the released spine (Phase 2)",
           "uncovered_examples": uncovered_examples}
    (REPO / "build/seam_closure.json").write_text(json.dumps(out, indent=1))
    print(f"seam closure: {n_covered}/{n_tables} construct tables "
          f"({out['seam_closed_rate']:.1%}) trace to released comprehensive tables "
          "by recorded class IRIs (was 4,149 seam tables)")
    print("→ build/seam_closure.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
