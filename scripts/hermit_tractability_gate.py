"""hermit_tractability_gate.py — STRUCTURAL HermiT-tractability pre-flight.

Assert tractability from the DL construct profile, rather than discover it as a 70-minute OOM on the release
artifact. The failure modes are well-characterized (Motik/Shearer/Horrocks, hypertableau/HermiT paper —
build/arxiv-1401.3485.md); each maps to specific OWL constructs the generator either emits or imports:

  #1 nominal + inverse + cardinality in JOINT scope → NI-rule promotes blockable individuals to unblockable
     ROOTS → doubly-exponential root individuals (§5.4). ── HARD-FAIL.
  #3 large n in `≤ n R.C` clausifies to O(n²) literals; counts belong in data, not TBox. ── budget n ≤ MAX_CARD.
  #2 existential cycles (C ⊑ ∃R.C) with rich labels → trees grow until pairwise blocking fires (2^(2m+2n)
     label space); worse under covering disjunctions. ── flag TBox ∃-cycles.
  #4 inverse roles at all → pairwise (not single) blocking, larger models. ── flag.
  #5 transitive roles under universals → concept-closure expansion in preprocessing. ── flag.

Reads OWL/TTL via rdflib (NO JVM). Emits the construct profile + DL expressivity + a verdict; exits non-zero
on any HARD-FAIL so it can gate a release. TBox is analyzed on its own (instance data must not ride along into
classification — verify the ABox separately).

  uv run --no-sync python scripts/hermit_tractability_gate.py corpora/ontology/sdg-ontology.owl [--json]
"""
from __future__ import annotations

import argparse
import json

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

MAX_CARD = 3          # ≤ n R.C clausifies to O(n²); keep small (#3)
MAX_DISJUNCTS = 12    # covering-disjunction fan-out over ∃-cycles is the K₂/K₁₁ blowup (#2)


def _exists_edges(g: rdflib.Graph) -> dict:
    """C --∃--> D whenever C (SubClassOf|≡-intersection-member) a restriction with someValuesFrom D."""
    owner: dict = {}
    for restr, _, filler in g.triples((None, OWL.someValuesFrom, None)):
        if not isinstance(filler, URIRef):
            continue
        for c, _, _r in g.triples((None, RDFS.subClassOf, restr)):
            if isinstance(c, URIRef):
                owner.setdefault(c, set()).add(filler)
        for c, _, eq in g.triples((None, OWL.equivalentClass, None)):
            lst = g.value(eq, OWL.intersectionOf)
            if lst is not None and isinstance(c, URIRef) and restr in list(Collection(g, lst)):
                owner.setdefault(c, set()).add(filler)
    return owner


def _in_cycle(edges: dict) -> set:
    """Nodes that can reach themselves through ∃-edges (participate in an existential cycle)."""
    hits = set()
    for start in edges:
        seen, stack = set(), list(edges.get(start, ()))
        while stack:
            x = stack.pop()
            if x == start:
                hits.add(start)
                break
            if x not in seen:
                seen.add(x)
                stack.extend(edges.get(x, ()))
    return hits


def _dl(nom: int, inv: int, trans: int, card: int) -> str:
    base = "S" if trans else "ALC"          # S = ALC + role transitivity
    s = base + "H"                          # H = role hierarchy (subPropertyOf; assumed present)
    if nom:
        s += "O"
    if inv:
        s += "I"
    if card:
        s += "Q"
    return s


def profile(path: str) -> dict:
    g = rdflib.Graph()
    g.parse(path)

    def npred(p) -> int:
        return sum(1 for _ in g.triples((None, p, None)))

    def ntype(t) -> int:
        return sum(1 for _ in g.triples((None, RDF.type, t)))

    nominals = npred(OWL.hasValue) + npred(OWL.oneOf)
    inverse = npred(OWL.inverseOf)  # covers both the InverseObjectProperties axiom and inverse expressions
    ifp = ntype(OWL.InverseFunctionalProperty)
    fp = ntype(OWL.FunctionalProperty)
    trans = ntype(OWL.TransitiveProperty)
    card_preds = [OWL.minCardinality, OWL.maxCardinality, OWL.cardinality,
                  OWL.minQualifiedCardinality, OWL.maxQualifiedCardinality, OWL.qualifiedCardinality]
    card_vals = [int(str(o)) for p in card_preds for _, _, o in g.triples((None, p, None))
                 if str(o).lstrip("-").isdigit()]
    n_card = len(card_vals) + ifp + fp        # IFP/FP are implicit ≤1 (cardinality on an inverse / functional)
    max_n = max(card_vals + ([1] if (ifp or fp) else []) + [0])
    disj = npred(OWL.unionOf) + npred(OWL.disjointUnionOf)
    exis = npred(OWL.someValuesFrom)
    cyc = _in_cycle(_exists_edges(g))

    # #1 the triad — nominal ∧ (inverse | IFP) ∧ cardinality in joint scope
    triad = nominals > 0 and (inverse + ifp) > 0 and n_card > 0
    fails, warns = [], []
    if triad:
        fails.append("TRIAD (#1): nominal + inverse + cardinality co-occur → NI-rule → unblockable roots")
    if max_n > MAX_CARD:
        fails.append(f"cardinality n={max_n} > {MAX_CARD} (#3): O(n²) clausification — counts belong in data")
    if (inverse + ifp) > 0:
        warns.append(f"{inverse + ifp} inverse role(s) (#4): pairwise blocking, larger models")
    if trans > 0:
        warns.append(f"{trans} transitive role(s) (#5): concept-closure expansion")
    if cyc:
        warns.append(f"{len(cyc)} class(es) in ∃-restriction cycles (#2): trees deepen until pairwise blocking")
    if cyc and disj > MAX_DISJUNCTS:
        fails.append(f"∃-cycles ∧ {disj} disjunctions (> {MAX_DISJUNCTS}) (#2): K₂/K₁₁ exponential-tree pattern")

    dl = _dl(nominals, inverse + ifp, trans, n_card)
    horn_ish = (nominals == 0 and (inverse + ifp) == 0 and max_n <= 1 and disj == 0)
    return {
        "path": path,
        "counts": {"nominals": nominals, "inverse": inverse, "inverse_functional": ifp, "functional": fp,
                   "transitive": trans, "cardinality_axioms": n_card, "max_cardinality_n": max_n,
                   "disjunctions": disj, "existentials": exis, "exists_cycle_classes": len(cyc)},
        "dl_expressivity": dl,
        "deterministic_fragment": "Horn-ish (no nominals/inverse, ≤1 card, no ⊔)" if horn_ish else dl,
        "triad": triad,
        "pass": not fails,
        "hard_fails": fails,
        "warnings": warns,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("owl", help="OWL/TTL file to profile (analyze the TBox on its own)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    p = profile(a.owl)
    if a.json:
        print(json.dumps(p, indent=1))
    else:
        c = p["counts"]
        print(f"DL={p['dl_expressivity']}  ({p['deterministic_fragment']})")
        print(f"  nominals={c['nominals']} inverse={c['inverse']}+{c['inverse_functional']}IFP "
              f"transitive={c['transitive']} card={c['cardinality_axioms']}(max n={c['max_cardinality_n']}) "
              f"⊔={c['disjunctions']} ∃={c['existentials']} ∃-cycle-classes={c['exists_cycle_classes']}")
        for f in p["hard_fails"]:
            print(f"  ✘ HARD-FAIL {f}")
        for w in p["warnings"]:
            print(f"  ⚠ {w}")
        print(f"  VERDICT: {'🟢 tractable-by-construction' if p['pass'] else '🔴 HARD-FAIL'}")
    return 0 if p["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
