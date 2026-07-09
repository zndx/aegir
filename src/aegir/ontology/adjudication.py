"""Sibling adjudication — the artifact, the universe rule, and the shared loaders (RH 2026-07-09).

The metric objective: every ADJUDICABLE sibling pair is ADJUDICATED — either
``disjoint`` (realized as a Manchester ``DisjointClasses`` axiom, HermiT-enforced,
kvasir-legible) or explicitly ``overlap`` (recorded; no logical claim). This supersedes
the naive sibling_disjointness→1.0 objective: blanket disjointness is WRONG where
siblings legitimately co-apply (roles co-inhere — one bearer may hold two roles).

THE UNIVERSE RULE (what counts as a sibling pair worth adjudicating): two sdg classes
are adjudicable siblings iff they share a SPECIFIC genus — a CCO or sdg class, or a BFO
*realizable* (role 0000023 / disposition 0000016 / realizable 0000017, where
co-membership semantics genuinely matter). Classes sharing only a bare BFO category
(continuant, occurrent, process, material entity, …) are NOT siblings — they are
GROUNDING DEBT (the ``bfo_grounded`` lever): the discriminating move there is grounding
to a specific genus first, then adjudicating within the real family. The excluded mass
is reported, never hidden.

Artifact: ``src/aegir/ontology/catalog/adjudications.json`` — per-family records
``{genus, default, disjoint_pairs[{a, b, rationale}], provenance}``. The realizer emits
``DisjointClasses`` from it (inside the HermiT boundary, so every future realize
re-validates every adjudication against the full ontology + ABox); the metrology counts
coverage from it; the closed loop (scripts/adjudicate_siblings.py) writes it.
"""
from __future__ import annotations

import json
from pathlib import Path

from rdflib import OWL, RDF, RDFS, URIRef

SDG = "https://signals.zndx.org/sdg#"
BFO_NS = "http://purl.obolibrary.org/obo/BFO_"
# BFO realizables — families where co-membership is meaningful; adjudicable by
# nomination-over-overlap-default (roles co-inhere; disjointness is the exception).
REALIZABLES = {f"{BFO_NS}{c}" for c in ("0000023", "0000016", "0000017")}

ADJUDICATIONS_FILE = Path(__file__).resolve().parent / "catalog" / "adjudications.json"


def first_named_genus(g, c):
    """The class's genus: the first named operand of its EquivalentTo intersection,
    else its first named superclass. Mirrors how the realizer mints structure."""
    for eq in g.objects(c, OWL.equivalentClass):
        for inter in g.objects(eq, OWL.intersectionOf):
            for item in g.items(inter):
                if isinstance(item, URIRef):
                    return item
        if isinstance(eq, URIRef):
            return eq
    for p in g.objects(c, RDFS.subClassOf):
        if isinstance(p, URIRef):
            return p
    return None


def genus_is_specific(genus: URIRef) -> bool:
    """The universe rule: CCO/sdg genera are specific; BFO genera are specific only
    when realizable. Bare BFO categorials = grounding debt, not siblinghood."""
    s = str(genus)
    if s.startswith(BFO_NS):
        return s in REALIZABLES
    return True   # cco:/sdg:/fhir-derived named genera


def sibling_families(owl_path: "str | Path | None" = None, *, graph=None) -> dict:
    """Compute the adjudication universe from the realized OWL (path or a pre-parsed
    rdflib graph — the metrology reuses its own).

    Returns ``{"adjudicable": {genus_str: [class_local, …]}, "debt": {genus_str: n_pairs},
    "universe_pairs": int, "debt_pairs": int}`` — local names (the ``sdg#`` fragment)
    throughout, matching the adjudication artifact and the Manchester render."""
    if graph is not None:
        g = graph
    else:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(owl_path))
    sdg = [c for c in g.subjects(RDF.type, OWL.Class)
           if isinstance(c, URIRef) and str(c).startswith(SDG)]
    fam: dict[URIRef, list[str]] = {}
    for c in sdg:
        genus = first_named_genus(g, c)
        if genus is not None:
            fam.setdefault(genus, []).append(str(c).split("#")[-1])
    adjudicable: dict[str, list[str]] = {}
    debt: dict[str, int] = {}
    for genus, kids in fam.items():
        if len(kids) < 2:
            continue
        n_pairs = len(kids) * (len(kids) - 1) // 2
        if genus_is_specific(genus):
            adjudicable[str(genus)] = sorted(kids)
        else:
            debt[str(genus)] = n_pairs
    return {"adjudicable": adjudicable, "debt": debt,
            "universe_pairs": sum(len(k) * (len(k) - 1) // 2 for k in adjudicable.values()),
            "debt_pairs": sum(debt.values())}


def load_adjudications(path: "str | Path | None" = None) -> dict:
    """The artifact, graceful-empty: ``{"families": {genus: record}}``."""
    p = Path(path) if path else ADJUDICATIONS_FILE
    if not p.exists():
        return {"version": "0.1", "families": {}}
    return json.loads(p.read_text())


def save_adjudications(data: dict, path: "str | Path | None" = None) -> Path:
    p = Path(path) if path else ADJUDICATIONS_FILE
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return p


def coverage(universe: dict, adj: dict) -> dict:
    """Adjudication coverage over the universe + the disjoint share sub-stat."""
    fams = adj.get("families", {})
    covered = disjoint = 0
    for genus, kids in universe["adjudicable"].items():
        n_pairs = len(kids) * (len(kids) - 1) // 2
        rec = fams.get(genus)
        if not rec:
            continue
        members = set(rec.get("members", []))
        if set(kids) <= members:          # the record covers the CURRENT family
            covered += n_pairs
            disjoint += len(rec.get("disjoint_pairs", []))
        # a stale record (family grew since adjudication) covers nothing — re-adjudicate
    up = universe["universe_pairs"]
    return {"sibling_pairs_universe": up,
            "sibling_pairs_grounding_debt": universe["debt_pairs"],
            "sibling_adjudication": round(covered / up, 4) if up else 1.0,
            "sibling_disjointness": round(disjoint / up, 4) if up else 0.0,
            "n_disjoint_axioms": disjoint}


def disjoint_manchester_frames(adj: dict) -> "list[str]":
    """The kvasir-legible Manchester lines the realizer appends — one
    ``DisjointClasses:`` frame per adjudicated-disjoint pair (sdg-prefixed)."""
    out = []
    for rec in adj.get("families", {}).values():
        for pair in rec.get("disjoint_pairs", []):
            out.append(f"DisjointClasses: sdg:{pair['a']}, sdg:{pair['b']}")
    return out
