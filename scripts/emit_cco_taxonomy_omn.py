#!/usr/bin/env python
"""emit_cco_taxonomy_omn — the imported theory's told taxonomy, as OMN text.

HermiT resolves the union's `Import: <cco-module.ttl>`; text-tier consumers (kvasir)
do not follow imports, so the CCO-side category lift (ont00000995 ⊑ bfo:0000004, module
disjointness, …) is invisible to them — the armed sweep judged with half the theory and
went clean on documents HermiT refused. This emits the module's TOLD subclass edges and
pairwise disjointness as plain OMN frames (sound: told axioms only, no inference), so
the union text is self-contained for any text-tier consumer.

    uv run python scripts/emit_cco_taxonomy_omn.py
    → build/grounding/cco-taxonomy.omn (consumed by realize_sdg --inline-theory)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CCO_TTL = REPO / "build/grounding/cco-module.ttl"
OUT = REPO / "build/grounding/cco-taxonomy.omn"

_PFX = [
    (re.compile(r"^http://purl\.obolibrary\.org/obo/BFO_(\d{7})$"), r"bfo:\1"),
    (re.compile(r"^https://www\.commoncoreontologies\.org/(ont\d+)$"), r"cco:\1"),
]


def pfx(iri: str) -> "str | None":
    for rx, rep in _PFX:
        if rx.match(iri):
            return rx.sub(rep, iri)
    return None


def main() -> int:
    import rdflib
    from rdflib.namespace import OWL, RDFS

    g = rdflib.Graph()
    g.parse(str(CCO_TTL))
    subs: "set[tuple]" = set()
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            ps, po = pfx(str(s)), pfx(str(o))
            if ps and po and ps != po:
                subs.add((ps, po))
    dis: "set[frozenset]" = set()
    for s, o in g.subject_objects(OWL.disjointWith):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            ps, po = pfx(str(s)), pfx(str(o))
            if ps and po and ps != po:
                dis.add(frozenset((ps, po)))

    lines = []
    for c, p in sorted(subs):
        lines.append(f"Class: {c}\n    SubClassOf: {p}\n")
    for pair in sorted(dis, key=sorted):
        a, b = sorted(pair)
        lines.append(f"DisjointClasses: {a}, {b}\n")
    OUT.write_text("\n".join(lines))
    print(f"{len(subs)} told subclass edges · {len(dis)} disjoint pairs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
