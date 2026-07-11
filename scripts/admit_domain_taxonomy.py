#!/usr/bin/env python
"""HermiT-admit the candidate domain taxonomy (Path A domain-taxonomy enrichment, stage 2).

``derive_domain_taxonomy.py`` proposes ``hypernym subClassOf <BFO anchor>`` + ``member subClassOf hypernym``
edges. This stage renders them — together with the BFO/CCO grounding (``Occurrent ⊥ Continuant``, the 7
anchors beneath) — into one Manchester ontology and classifies it with **HermiT**. A concept is REJECTED iff
it is **unsatisfiable**, which occurs exactly when it is grouped (across clusters) under both an
Occurrent-anchored and a Continuant-anchored hypernym — the genuine BFO category error. Consistent edges are
admitted to ``build/domain_taxonomy_admitted.json`` for emission into the SKOS vocab.

(Self-contained check: taxonomy + grounding. A richer pass that also loads each member's EXISTING
template-anchor assertion — catching a member grounded against its established anchor — is a future
enhancement.)

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/admit_domain_taxonomy.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_CAND = REPO / "build" / "domain_taxonomy_candidates.json"
_ADMITTED = REPO / "build" / "domain_taxonomy_admitted.json"

_ANCHOR_IRI = {
    "process": "bfo:0000015",
    "independent_continuant": "bfo:0000004",
    "artifact": "cco:ont00000995",
    "information_content_entity": "cco:ont00000958",
    "descriptive_ice": "cco:ont00000853",
    "directive_ice": "cco:ont00000965",
    "designative_ice": "cco:ont00000686",
}
_GROUNDING = """
Class: bfo:0000003
Class: bfo:0000002
DisjointClasses: bfo:0000003, bfo:0000002
Class: bfo:0000015 SubClassOf: bfo:0000003
Class: bfo:0000004 SubClassOf: bfo:0000002
Class: cco:ont00000995 SubClassOf: bfo:0000004
Class: cco:ont00000958 SubClassOf: bfo:0000002
Class: cco:ont00000853 SubClassOf: cco:ont00000958
Class: cco:ont00000965 SubClassOf: cco:ont00000958
Class: cco:ont00000686 SubClassOf: cco:ont00000958
"""
_HEADER = ("Prefix: bfo: <http://x/bfo#>\nPrefix: cco: <http://x/cco#>\nPrefix: dom: <http://x/dom#>\n"
           "Ontology: <http://x/aegir-domtax-admit>\n")


def _ident(c: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", c)
    return s if re.match(r"^[A-Za-z_]", s) else "c_" + s


def build_omn(taxa: "list[dict]") -> "tuple[str, dict[str, str]]":
    """Manchester doc = grounding + hypernym/member SubClassOf axioms. Returns (omn, dom-iri → concept)."""
    lines, iri2concept = [], {}
    for t in taxa:
        hyp = _ident(t["hypernym"])
        lines.append(f"Class: dom:{hyp} SubClassOf: {_ANCHOR_IRI[t['anchor']]}")
        iri2concept[f"http://x/dom#{hyp}"] = t["hypernym"]
        for m in t["members"]:
            mi = _ident(m)
            lines.append(f"Class: dom:{mi} SubClassOf: dom:{hyp}")
            iri2concept[f"http://x/dom#{mi}"] = m
    return _HEADER + _GROUNDING + "\n" + "\n".join(lines) + "\n", iri2concept


def main() -> int:
    from aegir.ontology.deeponto_harness import ensure_jvm
    cand = json.loads(_CAND.read_text())
    taxa = cand["taxa"]
    omn, iri2concept = build_omn(taxa)
    ensure_jvm()
    from deeponto.onto import Ontology
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".omn", delete=False) as f:
            f.write(omn)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        r = onto.reasoner.owl_reasoner
        consistent = bool(r.isConsistent())
        unsat = set()
        if consistent:
            bottom = r.getUnsatisfiableClasses()
            unsat = {str(c.getIRI()) for c in bottom.getEntities().toArray() if not c.isOWLNothing()}
    finally:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)

    bad_concepts = {iri2concept[i] for i in unsat if i in iri2concept}
    admitted, rejected = [], []
    for t in taxa:
        if t["hypernym"] in bad_concepts:
            rejected.append({**t, "reason": "hypernym unsatisfiable"})
            continue
        kept = [m for m in t["members"] if m not in bad_concepts]
        dropped = [m for m in t["members"] if m in bad_concepts]
        if len(kept) >= 2:
            admitted.append({"hypernym": t["hypernym"], "anchor": t["anchor"], "members": kept,
                             **({"dropped_members": dropped} if dropped else {})})
        else:
            rejected.append({**t, "reason": f"<2 satisfiable members (dropped {dropped})"})

    _ADMITTED.write_text(json.dumps(
        {"consistent": consistent, "n_hypernyms_in": len(taxa), "n_admitted": len(admitted),
         "n_rejected": len(rejected), "n_unsat_concepts": len(bad_concepts),
         "n_edges": sum(len(t["members"]) for t in admitted), "admitted": admitted, "rejected": rejected},
        indent=1))
    print(f"HermiT admission: consistent={consistent} | hypernyms {len(taxa)}→{len(admitted)} admitted "
          f"({len(rejected)} rejected) | {len(bad_concepts)} unsat concepts | "
          f"{sum(len(t['members']) for t in admitted)} edges → {_ADMITTED.relative_to(REPO)}", flush=True)
    for t in admitted[:12]:
        print(f"  ✓ {t['hypernym']} <- {t['anchor']}: {' | '.join(t['members'][:6])}", flush=True)
    for t in rejected[:6]:
        print(f"  ✗ {t['hypernym']}: {t['reason']}", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
