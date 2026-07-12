"""grounding — LOGICAL BFO/CCO grounding, computed by the reasoner (RH 2026-07-11).

Grounding is a LOGICAL property, not a syntactic one: a class is grounded iff the reasoner
ENTAILS it subsumed by some BFO or CCO class — through ANY chain (subClassOf, ≡, property
restrictions, imported π(CCO) axioms), including the nth-order chains the agent-mediated lexicon
earns through metric-guided refinement. The rdflib edge-walk in `ontology_metrology` only sees
first-order syntactic edges and undercounts this; the mandate ([[bfo_cco_grounding_mandate]])
requires the LOGIC.

This computes a per-class grounding certificate with the reasoner over the realized TBox (fast:
class-hierarchy classification only, no ABox — ~3s), which the metrology then READS (so the
OQuaRE gate needs no reasoner inline). PRE-KVASIR: the reasoner here is HermiT/JVM; per
[[kvasir_expensive_ontology_ops]] this entailment capability must move into kvasir (Rust) for
performance — the certificate contract stays the same across engines.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

SDG = "https://signals.zndx.org/sdg#"
BFO = "http://purl.obolibrary.org/obo/BFO_"
CCO = "https://www.commoncoreontologies.org/"

# the SKOS upper anchors as REAL BFO/CCO IRIs, most-specific first — a class's SKOS parent is the first
# anchor the reasoner entails it under (so build_skos_vocab derives the hierarchy from the OWL logic).
_ANCHOR_ORDER = [
    (CCO + "ont00000853", "SDG.ICE.DESCRIPTIVE"),
    (CCO + "ont00000965", "SDG.ICE.DIRECTIVE"),
    (CCO + "ont00000686", "SDG.ICE.DESIGNATIVE"),
    (CCO + "ont00000958", "SDG.ICE"),
    (BFO + "0000031", "SDG.GDC"),                 # generically dependent continuant (non-ICE)
    (BFO + "0000023", "SDG.ROLE"),                # realizable specifically-dependent continuants
    (BFO + "0000016", "SDG.DISPOSITION"),
    (BFO + "0000019", "SDG.QUALITY"),
    (BFO + "0000040", "SDG.MATERIAL_ENTITY"),     # under independent continuant
    (BFO + "0000015", "SDG.PROCESS"),
    (BFO + "0000004", "SDG.INDEPENDENT_CONTINUANT"),
]


def compute_grounding(omn_path: "str | Path") -> dict:
    """Reasoner-entailed BFO/CCO grounding per sdg class. Returns
    ``{"grounded": [locals], "ungrounded": [locals], "n": int, "rate": float, "engine": "hermit"}``.

    TBox-only (ABox stripped) — grounding is a class-subsumption fact and the ABox pass is the
    pathological cost we avoid ([[greenfield_reasoner_direction]]). A class counts as grounded iff
    an inferred super-or-equivalent class IRI is under the BFO or CCO namespace."""
    from aegir.ontology.deeponto_harness import ensure_jvm
    ensure_jvm()
    import jpype
    from deeponto.onto import Ontology

    lines = Path(omn_path).read_text().splitlines()
    first_ind = next((i for i, ln in enumerate(lines) if ln.startswith("Individual:")), len(lines))
    tbox = "\n".join(lines[:first_ind]).rstrip() + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(tbox)
        path = f.name

    onto = Ontology(path, reasoner_type="hermit")
    r = onto.reasoner.owl_reasoner
    InferenceType = jpype.JClass("org.semanticweb.owlapi.reasoner.InferenceType")
    r.precomputeInferences(jpype.JArray(InferenceType)([InferenceType.CLASS_HIERARCHY]))

    def _supers(c) -> set:
        out = {str(s.getIRI()) for s in r.getSuperClasses(c, False).getFlattened().toArray()}
        out |= {str(s.getIRI()) for s in r.getEquivalentClasses(c).getEntities().toArray()}
        return out

    # INTERNAL reasoning artifacts (ABox conjunction probes) are not real ontology classes — they must never
    # count toward grounding (they'd deflate the honest rate; a leaked run showed 84 probes dragging 0.957→0.850).
    classes = [c for c in onto.owl_onto.getClassesInSignature().toArray()
               if str(c.getIRI()).startswith(SDG) and "__conjprobe_" not in str(c.getIRI())]
    grounded, ungrounded, anchors = [], [], {}
    for c in classes:
        local = str(c.getIRI()).split("#")[-1]
        sup = _supers(c)
        if any(s.startswith(BFO) or s.startswith(CCO) for s in sup):
            grounded.append(local)
            # the most-specific SKOS anchor the reasoner ENTAILS membership in — so the SKOS hierarchy
            # can be derived FROM the logic (OWL ⊨ SKOS by construction), not a sparse template annotation.
            anchors[local] = next((code for iri, code in _ANCHOR_ORDER if iri in sup), "SDG.GENERIC")
        else:
            ungrounded.append(local)
    n = max(1, len(classes))
    return {"engine": "hermit", "n": len(classes),
            "grounded_count": len(grounded), "rate": round(len(grounded) / n, 4),
            "grounded": sorted(grounded), "ungrounded": sorted(ungrounded), "anchors": anchors}


def write_certificate(omn_path: "str | Path", out_path: "str | Path") -> dict:
    cert = compute_grounding(omn_path)
    Path(out_path).write_text(json.dumps(cert, indent=1))
    return cert


def load_certificate(path: "str | Path") -> "dict | None":
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
