#!/usr/bin/env python
"""Ontology metrology — a quantitative QUALITY PROFILE for an OWL ontology, benchmarked against the
IOF/BFO signature (Smith et al. 2019, "A First-Order Logic Formalization of the IOF Signature Using BFO").

Moves ontology QA from the binary "HermiT-consistent?" to a profile along the dimensions the IOF
formalization shows distinguish a *rigorous* BFO ontology from a merely-consistent one:

  1. DEFINITIONAL COMPLETENESS — fraction of classes with an equivalentClass (≡, necessary+sufficient,
     a genuine *definition*) vs subClassOf-only (→, necessary conditions = a *primitive*). The IOF
     defines ~half its terms with biconditionals; a generative ontology that only emits subClassOf
     restrictions scores ~0 here — the highest-leverage gap.
  2. BFO-GROUNDING DISCIPLINE — fraction of domain classes whose subsumption chain reaches a BFO
     category, and the distribution over BFO anchors.
  3. AXIOM EXPRESSIVITY — the DL-construct profile (∃ some / ∀ only / cardinality / disjoint / ≡).
  4. RELATIONAL RICHNESS (OntoQA) — existential restrictions per class; object-property inventory.
  5. BFO REALIZABLE-ENTITY MACHINERY — usage of role/disposition/function classes + realizes/inheres/
     bears/has-role/has-function properties. The IOF models phase-sortals (supplier, customer, raw
     material) as ROLES; a generative ontology that makes them rigid subclasses scores low here.

    uv run --no-sync python scripts/ontology_metrology.py corpora/ontology/sdg-ontology.owl
"""
from __future__ import annotations

import sys
from collections import Counter

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef

BFO = "http://purl.obolibrary.org/obo/BFO_"
SDG = "https://signals360.example.org/sdg#"
# BFO realizable-entity backbone (the IOF's modeling discipline)
REALIZABLE_CLASSES = {"0000023": "role", "0000016": "disposition", "0000034": "function", "0000017": "realizable entity"}
REALIZABLE_PROPS = {"0000055": "realizes", "0000052": "inheres in", "0000053": "bearer of",
                    "0000054": "realized in", "0000017b": "has role"}
# IOF signature benchmark (counted from Smith et al. 2019, §2.2.1–2.2.66)
IOF = {"terms": 66, "defin_complete": 0.55, "realizable_classes": 14, "object_properties": 21,
       "note": "FOL (beyond OWL-DL): ternary + temporally-indexed relations"}


def loc(iri: str) -> str:
    return iri.replace(BFO, "bfo:").split("#")[-1].split("/")[-1]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "corpora/ontology/sdg-ontology.owl"
    g = rdflib.Graph()
    g.parse(path)

    named = [c for c in set(g.subjects(RDF.type, OWL.Class)) if isinstance(c, URIRef)]
    sdg = [c for c in named if str(c).startswith(SDG)]
    obj_props = [p for p in set(g.subjects(RDF.type, OWL.ObjectProperty)) if isinstance(p, URIRef)]
    sdg_props = [p for p in obj_props if str(p).startswith(SDG)]

    # 1. definitional completeness
    defined = {c for c in sdg if (c, OWL.equivalentClass, None) in g}

    # 2. BFO grounding (subClassOf chain reaching a BFO IRI)
    parents: dict = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            parents.setdefault(s, []).append(o)

    def anchor(c, seen=None):
        seen = seen or set()
        for p in parents.get(c, []):
            if str(p).startswith(BFO):
                return p
            if p not in seen:
                seen.add(p)
                a = anchor(p, seen)
                if a:
                    return a
        return None

    anchors = {c: anchor(c) for c in sdg}
    grounded = [c for c, a in anchors.items() if a]
    bfo_dist = Counter(loc(str(a)) for a in anchors.values() if a)

    # 3. axiom expressivity
    n_sub = sum(1 for s, _, _o in g.triples((None, RDFS.subClassOf, None)) if isinstance(s, URIRef))
    n_equiv = len(list(g.triples((None, OWL.equivalentClass, None))))
    n_some = len(list(g.subjects(OWL.someValuesFrom, None)))
    n_all = len(list(g.subjects(OWL.allValuesFrom, None)))
    n_card = sum(len(list(g.subjects(p, None))) for p in
                 (OWL.cardinality, OWL.minCardinality, OWL.maxCardinality,
                  OWL.qualifiedCardinality, OWL.minQualifiedCardinality, OWL.maxQualifiedCardinality))
    n_disj = len(list(g.triples((None, OWL.disjointWith, None)))) + len(list(g.subjects(RDF.type, OWL.AllDisjointClasses)))

    # 5. realizable-machinery usage
    realiz_cls = {URIRef(BFO + k) for k in REALIZABLE_CLASSES}
    realiz_prop = {URIRef(BFO + k) for k in ("0000055", "0000052", "0000053", "0000054")}
    on_props = Counter(str(o) for _, _, o in g.triples((None, OWL.onProperty, None)))
    svf = Counter(str(o) for _, _, o in g.triples((None, OWL.someValuesFrom, None)))
    realiz_prop_uses = sum(n for iri, n in on_props.items() if URIRef(iri) in realiz_prop)
    realiz_cls_uses = sum(n for iri, n in svf.items() if URIRef(iri) in realiz_cls)

    n = max(1, len(sdg))
    print(f"=== ONTOLOGY METROLOGY · {path.split('/')[-1]} ===")
    print(f"domain classes (sdg:): {len(sdg)}   object properties (sdg:): {len(sdg_props)}   "
          f"all named classes: {len(named)}")
    print()
    print("DIMENSION                         OURS            IOF/BFO benchmark")
    print(f"1 definitional completeness       {len(defined)/n:5.1%} ({len(defined)}/{len(sdg)})    ~{IOF['defin_complete']:.0%}  (≡ biconditional defs)")
    print(f"2 BFO-grounded                    {len(grounded)/n:5.1%} ({len(grounded)}/{len(sdg)})    100% (every term BFO-anchored)")
    print(f"3 relational richness (∃/class)   {n_some/n:5.2f}            rich (role/realizes patterns)")
    print(f"4 realizable-machinery usage      {realiz_prop_uses + realiz_cls_uses:<5}           {IOF['realizable_classes']}+ role/disp/fn classes")
    print()
    print("AXIOM EXPRESSIVITY PROFILE")
    print(f"  subClassOf (→ necessary):       {n_sub}")
    print(f"  equivalentClass (≡ defining):   {n_equiv}")
    print(f"  ∃ some (existential):           {n_some}")
    print(f"  ∀ only (universal):             {n_all}")
    print(f"  cardinality:                    {n_card}")
    print(f"  disjointness:                   {n_disj}")
    print()
    print("BFO ANCHOR DISTRIBUTION (top):", dict(bfo_dist.most_common(8)))
    print()
    print(f"IOF note: {IOF['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
