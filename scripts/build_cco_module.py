"""build_cco_module.py — π(CCO): the Horn projection of CCO over the fragment that constrains our domain.

CCO's VALUE to us is its subsumption taxonomy (grounds our cco:ont refs to BFO) + its class disjointness  # coined-ok: prose/placeholder, not a real ref
(catches a grounding landing under two disjoint branches). Both are Horn — deterministic, linear for the
hypertableau calculus. CCO's COST is its SHIQ machinery: 79 inverse + 8 transitive + 37 ⊔, over cco: PROPERTIES
and unions that our ontology never references (our roles are sdg:/bfo:, our grounding is subClassOf/≡-genus).

A ⊥-locality module was the first attempt — semantically lossless over our signature — but it faithfully
preserves 27 inverse roles, and 27 inverse × our 442 ≡ is a pairwise-blocking blowup: HermiT ran 1h18m without
terminating (the structural gate mispredicted it; the reasoner is ground truth). So we NARROW the theory to its
constraining Horn fragment rather than relax the check: keep subClassOf + disjointWith + AllDisjoint (expanded
to pairwise) + domain/range + declarations + labels; keep the named genus of each equivalentClass (as
subClassOf) so defined cco: classes still ground; drop inverse/transitive/symmetric/functional, ⊔, and
someValuesFrom restrictions. This CANNOT relax our domain's consistency — no grounding of ours can be refuted
via constructs it doesn't use — it drops only CCO-internal machinery. π(CCO) = BFO + this, the one fixed theory.

Pure rdflib (no JVM). Gate-certify with hermit_tractability_gate before it becomes load-bearing.

  uv run --no-sync python scripts/build_cco_module.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

REPO = Path(__file__).resolve().parent.parent
CCO = REPO / "build" / "grounding" / "cco-merged.ttl"
OUT = REPO / "build" / "grounding" / "cco-module.ttl"
SKOS_DEF = URIRef("http://www.w3.org/2004/02/skos/core#definition")
IAO_DEF = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
DECL = {OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty}


def _named_members(g: rdflib.Graph, node) -> "list[URIRef]":
    """Named classes of an intersection/union list (the genus/genera), or [node] if node is itself named."""
    if isinstance(node, URIRef):
        return [node]
    for lp in (OWL.intersectionOf, OWL.unionOf):
        lst = g.value(node, lp)
        if lst is not None:
            return [m for m in Collection(g, lst) if isinstance(m, URIRef)]
    return []


def horn_project(g: rdflib.Graph) -> rdflib.Graph:
    out = rdflib.Graph()
    for pfx, ns in g.namespaces():
        out.bind(pfx, ns)

    # subsumption taxonomy (named → named) — grounds our cco:ont refs up to BFO  # coined-ok: prose/placeholder, not a real ref
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            out.add((s, RDFS.subClassOf, o))
    # defined classes: keep the named genus as subClassOf (grounding), drop the existential/⊔ differentia (cost)
    for s, o in g.subject_objects(OWL.equivalentClass):
        if isinstance(s, URIRef):
            for m in _named_members(g, o):
                out.add((s, RDFS.subClassOf, m))

    # THE teeth — class disjointness (Horn: A ⊓ B ⊑ ⊥). AllDisjoint expanded to pairwise disjointWith.
    for s, o in g.subject_objects(OWL.disjointWith):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            out.add((s, OWL.disjointWith, o))
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        lst = g.value(adc, OWL.members)
        members = [m for m in Collection(g, lst) if isinstance(m, URIRef)] if lst is not None else []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                out.add((members[i], OWL.disjointWith, members[j]))

    # domain/range (Horn) + declarations (drop property-characteristic types) + labels/definitions
    for p in (RDFS.domain, RDFS.range):
        for s, o in g.subject_objects(p):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                out.add((s, p, o))
    for s, o in g.subject_objects(RDF.type):
        if o in DECL and isinstance(s, URIRef):
            out.add((s, RDF.type, o))
    for p in (RDFS.label, SKOS_DEF, IAO_DEF):
        for s, o in g.subject_objects(p):
            if isinstance(s, URIRef):
                out.add((s, p, o))
    return out


def main() -> int:
    g = rdflib.Graph()
    g.parse(str(CCO), format="turtle")
    m = horn_project(g)
    m.serialize(str(OUT), format="turtle")
    dj = len(list(m.triples((None, OWL.disjointWith, None))))
    sc = len(list(m.triples((None, RDFS.subClassOf, None))))
    cls = len({s for s in m.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)})
    print(f"π(CCO) Horn projection: {len(m)} triples · {cls} classes · {sc} subClassOf · {dj} disjointWith "
          f"(from CCO's {len(g)} triples) → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
