"""rigor_diagnose — enumerate the ontology's rigor-failing classes as a curation worklist.

The metrology (`ontology_metrology.compute`) reports only AGGREGATES; the failing-class sets are
closures inside it. This replicates its exact predicates (parents/anchor `:86-115`, defined `:83`,
has_def `:137-139`) to DUMP the per-class worklist a rigor-closing pass (and the autonomous
Qwen3.6 loop) needs:

  ungrounded  — anchor(c) is None  (== the orphans; no genus reaches a BFO IRI)   → bfo_grounded
  unannotated — no rdfs:comment / IAO_0000115 / skos:definition                    → def_annotation
  undefined   — no owl:equivalentClass axiom                                       → def_completeness

Each class is annotated with its current genus (or ORPHAN), the OntoClean `suggested` pattern
(role/equivalent_class/disposition/…), and whether it sits in a generic-BFO GROUNDING-DEBT family
(the specific-vs-generic depth signal the floors themselves miss). Prioritizes orphans, since one
specific genus closes grounding + orphan together. Writes build/rigor_worklist.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import rdflib  # noqa: E402
from rdflib.collection import Collection  # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS  # noqa: E402

from ontology_metrology import BFO, SDG, _cco_backbone, loc  # noqa: E402
from aegir.ontology import adjudication as ADJ  # noqa: E402
from aegir.ontology import ontoclean  # noqa: E402

IAO_DEF = rdflib.URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
SKOS_DEF = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#definition")

# gate floors this pass must clear
BFO_FLOOR = 0.95
ANN_FLOOR = 0.90


def _parents(g) -> dict:
    """subClassOf edges + named members of each ≡-intersection (the metrology's `parents`, :86-100)."""
    parents: dict = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            parents.setdefault(s, []).append(o)
    for s, _, eq in g.triples((None, OWL.equivalentClass, None)):
        if not isinstance(s, rdflib.URIRef):
            continue
        lst = g.value(eq, OWL.intersectionOf)
        if lst is not None:
            for member in Collection(g, lst):
                if isinstance(member, rdflib.URIRef):
                    parents.setdefault(s, []).append(member)
    return parents


def _anchor(c, parents, seen=None):
    seen = seen or set()
    for p in parents.get(c, []):
        if str(p).startswith(BFO):
            return p
        if p not in seen:
            seen.add(p)
            a = _anchor(p, parents, seen)
            if a:
                return a
    return None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "corpora" / "ontology" / "sdg-ontology.owl")
    g = rdflib.Graph()
    g.parse(path)
    for t in _cco_backbone():
        g.add(t)

    named = [c for c in set(g.subjects(RDF.type, OWL.Class)) if isinstance(c, rdflib.URIRef)]
    sdg = [c for c in named if str(c).startswith(SDG)]
    n = max(1, len(sdg))
    parents = _parents(g)

    def label(c):
        v = g.value(c, RDFS.label)
        return str(v) if v is not None else str(c).split("#")[-1]

    def genus(c):
        ps = [loc(str(p)) for p in parents.get(c, [])]
        return ps[0] if ps else "ORPHAN"

    ungrounded = [c for c in sdg if _anchor(c, parents) is None]
    unannotated = [c for c in sdg if not (
        (c, RDFS.comment, None) in g or (c, IAO_DEF, None) in g or (c, SKOS_DEF, None) in g)]
    undefined = [c for c in sdg if (c, OWL.equivalentClass, None) not in g]

    # grounding-debt families (generic-BFO genus siblings) — the specific-vs-generic depth signal.
    # sibling_families()["debt"] maps genus → n_pairs; we key debt membership by the genus local name.
    fams = ADJ.sibling_families(graph=g)
    debt_genera = {gg.split(":")[-1].split("#")[-1].split("/")[-1] for gg in (fams.get("debt") or {})}

    def suggest(c):
        man = str(g.value(c, RDFS.comment) or "")
        return ontoclean.classify(label(c), definition=man).get("suggested", "keep")

    def row(c):
        return {
            "iri": str(c),
            "local": str(c).split("#")[-1],
            "label": label(c),
            "genus": genus(c),
            "orphan": not parents.get(c),
            "suggested": suggest(c),
            "in_debt_family": genus(c) in debt_genera,
        }

    # how many fixes cross each floor
    n_grounded = n - len(ungrounded)
    n_annot = n - len(unannotated)
    need_ground = max(0, math.ceil(BFO_FLOOR * n) - n_grounded)
    need_annot = max(0, math.ceil(ANN_FLOOR * n) - n_annot)

    worklist = {
        "ontology": path,
        "n_domain_classes": n,
        "floors": {"bfo_grounded": BFO_FLOOR, "def_annotation_coverage": ANN_FLOOR},
        "current": {
            "bfo_grounded": round(n_grounded / n, 4),
            "def_annotation_coverage": round(n_annot / n, 4),
            "definitional_completeness": round((n - len(undefined)) / n, 4),
            "orphan_rate": round(len(ungrounded) / n, 4),
        },
        "to_close_floor": {"ground": need_ground, "annotate": need_annot},
        "grounding_debt_families": len(debt_genera),
        # orphans first (one specific genus closes grounding + orphan together)
        "ungrounded": sorted((row(c) for c in ungrounded), key=lambda r: (not r["orphan"], r["local"])),
        "unannotated": sorted((row(c) for c in unannotated), key=lambda r: r["local"]),
        "undefined": sorted((row(c) for c in undefined), key=lambda r: r["local"]),
    }
    out = REPO / "build" / "rigor_worklist.json"
    out.write_text(json.dumps(worklist, indent=1))

    print(f"=== rigor worklist ({path}) — {n} domain classes ===")
    print(f"  bfo_grounded              {worklist['current']['bfo_grounded']}  (floor {BFO_FLOOR}) "
          f"→ ground {need_ground} more of {len(ungrounded)} ungrounded")
    print(f"  def_annotation_coverage   {worklist['current']['def_annotation_coverage']}  (floor {ANN_FLOOR}) "
          f"→ annotate {need_annot} more of {len(unannotated)} unannotated")
    print(f"  definitional_completeness {worklist['current']['definitional_completeness']}  "
          f"({len(undefined)} undefined — depth headroom)")
    print(f"  grounding-debt families (generic-BFO genus): {len(debt_genera)}")
    from collections import Counter
    sug = Counter(r["suggested"] for r in worklist["ungrounded"])
    print(f"  ungrounded by OntoClean suggestion: {dict(sug)}")
    print(f"  → wrote {out.relative_to(REPO)}")
    print("\n  first 12 ungrounded (orphans first):")
    for r in worklist["ungrounded"][:12]:
        print(f"    {r['local']:42s} genus={r['genus']:20s} suggest={r['suggested']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
