#!/usr/bin/env python
"""Ontology metrology — a quantitative QUALITY PROFILE for an OWL ontology, benchmarked against the
IOF/BFO signature (Smith et al. 2019, "A First-Order Logic Formalization of the IOF Signature Using BFO").

Moves ontology QA from the binary "HermiT-consistent?" to a profile along the dimensions the IOF
formalization shows distinguish a *rigorous* BFO ontology from a merely-consistent one:

  1. DEFINITIONAL COMPLETENESS — fraction of classes with an equivalentClass (≡, necessary+sufficient,
     a genuine *definition*) vs subClassOf-only (→, necessary conditions = a *primitive*).
  2. BFO-GROUNDING DISCIPLINE — fraction of domain classes whose subsumption chain reaches a BFO category.
  3. AXIOM EXPRESSIVITY — the DL-construct profile (∃ some / ∀ only / cardinality / disjoint / ≡).
  4. RELATIONAL RICHNESS (OntoQA) — existential restrictions per class; object-property inventory.
  5. BFO REALIZABLE-ENTITY MACHINERY — usage of role/disposition/function classes + realizes/inheres props.

`compute(path) -> dict` is the single source of truth consumed by the OQuaRE gate
(`scripts/ontology_oquare.py`) and the lineup loader (`aegir.lineup.sources.ontology_metrology`).

    uv run --no-sync python scripts/ontology_metrology.py corpora/ontology/sdg-ontology.owl [--json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef

BFO = "http://purl.obolibrary.org/obo/BFO_"
SDG = "https://signals360.example.org/sdg#"
# BFO realizable-entity backbone (the IOF's modeling discipline)
REALIZABLE_CLASSES = {"0000023": "role", "0000016": "disposition", "0000034": "function", "0000017": "realizable entity"}
REALIZABLE_PROPS = ("0000055", "0000052", "0000053", "0000054")  # realizes / inheres-in / bearer-of / realized-in
# IOF signature benchmark (counted from Smith et al. 2019, §2.2.1–2.2.66)
IOF = {"terms": 66, "defin_complete": 0.55, "realizable_classes": 14, "object_properties": 21,
       "note": "FOL (beyond OWL-DL): ternary + temporally-indexed relations"}


def loc(iri: str) -> str:
    return iri.replace(BFO, "bfo:").split("#")[-1].split("/")[-1]


def compute(path: str) -> dict:
    """The full ontology-quality profile for an OWL file — the single source of truth for the OQuaRE gate
    and the lineup. Pure rdflib (no JVM)."""
    g = rdflib.Graph()
    g.parse(path)

    named = [c for c in set(g.subjects(RDF.type, OWL.Class)) if isinstance(c, URIRef)]
    sdg = [c for c in named if str(c).startswith(SDG)]
    sdg_props = [p for p in set(g.subjects(RDF.type, OWL.ObjectProperty))
                 if isinstance(p, URIRef) and str(p).startswith(SDG)]
    dataprops = [p for p in set(g.subjects(RDF.type, OWL.DatatypeProperty)) if isinstance(p, URIRef)]

    # 1. definitional completeness
    defined = {c for c in sdg if (c, OWL.equivalentClass, None) in g}

    # 2. BFO grounding (subClassOf chain reaching a BFO IRI)
    parents: dict = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            parents.setdefault(s, []).append(o)
    # equivalentClass grounding: a defined class X ≡ (Genus ⊓ …restrictions…) is subsumed by the NAMED members
    # of the intersection (its genus) — the reasoner entails the subClassOf, so it counts for BFO-grounding.
    from rdflib.collection import Collection  # noqa: PLC0415
    for s, _, eq in g.triples((None, OWL.equivalentClass, None)):
        if not isinstance(s, URIRef):
            continue
        lst = g.value(eq, OWL.intersectionOf)
        if lst is not None:
            for member in Collection(g, lst):
                if isinstance(member, URIRef):
                    parents.setdefault(s, []).append(member)

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
    realiz_prop = {URIRef(BFO + k) for k in REALIZABLE_PROPS}
    on_props = Counter(str(o) for _, _, o in g.triples((None, OWL.onProperty, None)))
    svf = Counter(str(o) for _, _, o in g.triples((None, OWL.someValuesFrom, None)))
    realiz_uses = (sum(n for iri, n in on_props.items() if URIRef(iri) in realiz_prop)
                   + sum(n for iri, n in svf.items() if URIRef(iri) in realiz_cls))

    # def-annotation coverage (the IAO/OBO convention)
    iao_def = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
    skos_def = URIRef("http://www.w3.org/2004/02/skos/core#definition")
    has_def = {c for c in sdg if (c, RDFS.comment, None) in g or (c, iao_def, None) in g or (c, skos_def, None) in g}

    depth_cache: dict = {}

    def depth(c):  # longest subClassOf chain (DAG; temp-guard against cycles)
        if c in depth_cache:
            return depth_cache[c]
        depth_cache[c] = 1
        ps = [p for p in parents.get(c, []) if isinstance(p, URIRef)]
        depth_cache[c] = 1 + max((depth(p) for p in ps), default=0)
        return depth_cache[c]

    n = max(1, len(sdg))
    return {
        "path": path,
        "n_domain_classes": len(sdg), "n_object_properties": len(sdg_props),
        "n_datatype_properties": len(dataprops),
        # IOF-derived rigor dimensions
        "definitional_completeness": len(defined) / n,
        "bfo_grounded": len(grounded) / n,
        "realizable_machinery": realiz_uses,
        "def_annotation_coverage": len(has_def) / n,
        # field-standard structural metrics (OntoQA / OQuaRE)
        "rr": n_some / max(1, n_sub + n_some),
        "ir": n_sub / n,
        "ar": len(dataprops) / n,
        "aronto": (n_some + n_all + n_card) / n,
        "dit": max((depth(c) for c in sdg), default=0),
        "tm": sum(1 for c in sdg if len([p for p in parents.get(c, []) if isinstance(p, URIRef)]) > 1) / n,
        # axiom expressivity + anchors
        "n_subclass": n_sub, "n_equiv": n_equiv, "n_some": n_some, "n_all": n_all,
        "n_card": n_card, "n_disjoint": n_disj, "bfo_anchors": dict(bfo_dist.most_common(8)),
        # raw counts (for the OQuaRE module + diagnostics)
        "n_defined": len(defined), "n_grounded": len(grounded), "n_annotated": len(has_def),
    }


def _print_profile(m: dict) -> None:
    print(f"=== ONTOLOGY METROLOGY · {m['path'].split('/')[-1]} ===")
    print(f"domain classes (sdg:): {m['n_domain_classes']}   object properties: {m['n_object_properties']}   "
          f"datatype props: {m['n_datatype_properties']}")
    print()
    print("RIGOR DIMENSIONS  (IOF-derived — what the field's structural metrics MISS)   OURS      IOF/BFO")
    print(f"  definitional completeness  (≡ defs vs subClassOf primitives)             {m['definitional_completeness']:6.1%}   ~55%")
    print(f"  BFO-grounded               (subsumption chain reaches a BFO category)    {m['bfo_grounded']:6.1%}   100%")
    print(f"  realizable machinery       (role/disposition/function usage)             {m['realizable_machinery']:<6}   14+")
    print(f"  definition-annotation cov. (NL/FOL definitions, the IOF convention)      {m['def_annotation_coverage']:6.1%}   100% req")
    print()
    print("FIELD-STANDARD METRICS  (OntoQA / OQuaRE — auto-computable, comparable)      OURS")
    print(f"  RR  relationship richness  (rich beyond taxonomy; pure tree -> 0)        {m['rr']:6.2f}")
    print(f"  IR  inheritance richness   (subclasses per class)                        {m['ir']:6.2f}")
    print(f"  AR  attribute richness     (datatype props per class)                    {m['ar']:6.2f}")
    print(f"  AROnto axiomatic strength  (restrictions per class)                      {m['aronto']:6.2f}")
    print(f"  DITOnto max depth          / TMOnto tangledness                          {m['dit']:<3} / {m['tm']:.1%}")
    print()
    print(f"AXIOM EXPRESSIVITY: {m['n_subclass']} subClassOf · {m['n_equiv']} ≡ · {m['n_some']} ∃some · "
          f"{m['n_all']} ∀only · {m['n_card']} card · {m['n_disjoint']} disjoint")
    print(f"BFO ANCHORS (top): {m['bfo_anchors']}")
    print()
    print("READ: the FIELD-STANDARD metrics (RR/IR/AROnto) score us decently — they are blind to the")
    print("rigor that distinguishes IOF. The IOF-DERIVED dimensions (definitional completeness, realizable")
    print("machinery, definition coverage) are the discriminating add — and the generative deriver's roadmap.")


def main() -> int:
    ap = argparse.ArgumentParser(description="ontology metrology profile (IOF-benchmarked)")
    ap.add_argument("ontology", nargs="?", default="corpora/ontology/sdg-ontology.owl")
    ap.add_argument("--json", action="store_true", help="emit the metric dict as JSON")
    args = ap.parse_args()
    m = compute(args.ontology)
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        _print_profile(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
