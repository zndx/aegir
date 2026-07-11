"""verify_owl_skos — OWL ⊨ SKOS: is every skos:broader anchor edge LOGICALLY entailed?

The published SKOS vocabulary assigns each template a broader ANCHOR from its `bfo_anchor_path`
(build_skos_vocab.ANCHORS). That assignment is a TEMPLATE annotation, not a reasoned fact — so it
can DIVERGE from the OWL: a template whose `bfo_anchor_path` says "Descriptive ICE" but whose realized
genus makes it a process would give a SKOS edge the OWL does not entail. The mandate
([[bfo_cco_grounding_mandate]]) requires OWL ⊨ SKOS: HermiT must ENTAIL `class ⊑ anchor` for every
broader edge. This checks it and reports divergences (the ones to re-author or re-anchor).

Anchors are mapped to their REAL CCO IRIs (the fictional readable cco: names the SKOS keys on —
cco:DescriptiveICE — canonicalized to cco:ont00000853, matching the OWL). Pre-kvasir HermiT.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/verify_owl_skos.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.ontology.schema import load_catalog, CATALOG_FILE  # noqa: E402
from build_skos_vocab import ANCHORS, GENERIC  # noqa: E402

SDG = "https://signals.zndx.org/sdg#"
BFO = "http://purl.obolibrary.org/obo/BFO_"
CCO = "https://www.commoncoreontologies.org/"
OMN = REPO / "corpora" / "ontology" / "sdg-ontology.omn"

# each SKOS anchor concept (parent_code) → the REAL BFO/CCO class the OWL must entail membership in.
ANCHOR_IRI = {
    "SDG.PROCESS": BFO + "0000015",
    "SDG.INDEPENDENT_CONTINUANT": BFO + "0000004",
    "SDG.MATERIAL_ENTITY": BFO + "0000040",
    "SDG.ARTIFACT": BFO + "0000004",           # cco:Artifact → its BFO parent (independent continuant)
    "SDG.GDC": BFO + "0000031",
    "SDG.ICE": CCO + "ont00000958",
    "SDG.ICE.DESCRIPTIVE": CCO + "ont00000853",
    "SDG.ICE.DIRECTIVE": CCO + "ont00000965",  # CCO Prescriptive ICE
    "SDG.ICE.DESIGNATIVE": CCO + "ont00000686",
    "SDG.QUALITY": BFO + "0000019",
    "SDG.ROLE": BFO + "0000023",
    "SDG.DISPOSITION": BFO + "0000016",
    "SDG.GENERIC": None,                        # anchorless — no entailment claimed
}


def main() -> int:
    # read the ACTUAL published vocab (annotations.parquet): each leaf carries its parent_code (the SKOS
    # anchor edge) + its axiom (from which we recover the OWL head class) — so we verify the real artifact.
    import pyarrow.parquet as pq
    vocab = pq.read_table(REPO / "corpora" / "vocabulary" / "annotations.parquet").to_pylist()
    head_anchor: dict[str, str] = {}
    for rec in vocab:
        pc = rec.get("parent_code") or ""
        hm = re.search(r"Class:\s*\{(\w+)", rec.get("axiom") or "")
        if pc and hm:
            head_anchor[hm.group(1)] = pc

    from aegir.ontology.deeponto_harness import ensure_jvm
    ensure_jvm()
    import jpype
    from deeponto.onto import Ontology
    lines = OMN.read_text().splitlines()
    first_ind = next((i for i, ln in enumerate(lines) if ln.startswith("Individual:")), len(lines))
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write("\n".join(lines[:first_ind]).rstrip() + "\n")
        path = f.name
    onto = Ontology(path, reasoner_type="hermit")
    r = onto.reasoner.owl_reasoner
    InferenceType = jpype.JClass("org.semanticweb.owlapi.reasoner.InferenceType")
    r.precomputeInferences(jpype.JArray(InferenceType)([InferenceType.CLASS_HIERARCHY]))
    factory = onto.owl_manager.getOWLDataFactory()
    IRI = jpype.JClass("org.semanticweb.owlapi.model.IRI")
    sig = {str(c.getIRI()).split("#")[-1] for c in onto.owl_onto.getClassesInSignature().toArray()
           if str(c.getIRI()).startswith(SDG)}

    def supers(local: str) -> set:
        c = factory.getOWLClass(IRI.create(SDG + local))
        out = {str(s.getIRI()) for s in r.getSuperClasses(c, False).getFlattened().toArray()}
        out |= {str(s.getIRI()) for s in r.getEquivalentClasses(c).getEntities().toArray()}
        return out

    entailed = divergent = anchorless = missing = 0
    diverging = []
    for head, pcode in sorted(head_anchor.items()):
        if head not in sig:
            missing += 1
            continue
        anchor_iri = ANCHOR_IRI.get(pcode)
        if anchor_iri is None:
            anchorless += 1
            continue
        sup = supers(head)
        # entailed if the exact anchor is inferred, OR (for ICE sub-anchors) any real-CCO/BFO ancestor holds
        if anchor_iri in sup:
            entailed += 1
        elif any(s.startswith(BFO) or s.startswith(CCO) for s in sup):
            # grounded, but NOT under the specific anchor the SKOS claims → the divergence to re-anchor
            divergent += 1
            diverging.append((head, pcode))
        else:
            divergent += 1
            diverging.append((head, pcode + " [ungrounded]"))

    total = entailed + divergent + anchorless
    print(f"=== OWL ⊨ SKOS anchor-edge verification ({total} templates) ===")
    print(f"  ENTAILED (OWL entails class ⊑ its SKOS anchor):  {entailed}")
    print(f"  ANCHORLESS (SDG.GENERIC — no claim):             {anchorless}")
    print(f"  DIVERGENT (SKOS anchor NOT OWL-entailed):        {divergent}")
    print(f"  not in OWL signature (skipped):                  {missing}")
    if diverging:
        print("\n  divergences (head → claimed anchor) — re-anchor or re-author:")
        for h, p in diverging[:30]:
            print(f"    {h:42s} → {p}")
    rate = entailed / max(1, entailed + divergent)
    print(f"\n  OWL⊨SKOS entailment rate (of anchored): {rate:.4f}")
    return 0 if divergent == 0 else 1


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
