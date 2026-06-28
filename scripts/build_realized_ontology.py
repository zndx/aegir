#!/usr/bin/env python
"""Realize the FinePDFs-derived templates into a consumer-ready, HermiT-validated BFO/CCO-grounded
domain ontology, and emit it (+ a consistency certificate) into ``corpora/ontology/``.

The catalog ships *templates* (Manchester skeletons with the `{slot:Type}` DSL). This MATERIALIZES the
83 FinePDFs-derived templates (``08_derived``) into concrete OWL axioms over global ``sdg:`` domain IRIs
(so a class shared by two templates is ONE class), assembles them with the BFO/CCO grounding, validates
the result with HermiT, and emits the artifact a consumer can load and re-reason over:

    corpora/ontology/sdg-ontology.omn   the realized ontology (OWL Manchester syntax)
    corpora/ontology/sdg-ontology.owl   the same, RDF/XML (best-effort, for owlready2/Protégé)
    corpora/ontology/HERMIT_CERTIFICATE.md   the consistency certificate

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py
"""
from __future__ import annotations

import datetime
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import reasoning_gates as RG  # noqa: E402
from aegir.ontology.deeponto_harness import TEST_NAMESPACE, ensure_jvm  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

OUT = REPO / "corpora" / "ontology"
SDG_NS = "https://signals360.example.org/sdg#"

# Numeric BFO 2020 grounding so the derived axioms (which anchor to bfo:0000015 etc.) are reasoned
# MEANINGFULLY — the continuant⊥occurrent disjointness is what makes a cross-category head unsatisfiable.
NUMERIC_BFO = """
Class: bfo:0000001
Class: bfo:0000002 SubClassOf: bfo:0000001
Class: bfo:0000003 SubClassOf: bfo:0000001
DisjointClasses: bfo:0000002, bfo:0000003
Class: bfo:0000015 SubClassOf: bfo:0000003
Class: bfo:0000004 SubClassOf: bfo:0000002
Class: bfo:0000017 SubClassOf: bfo:0000002
Class: bfo:0000019 SubClassOf: bfo:0000020
Class: bfo:0000020 SubClassOf: bfo:0000002
Class: bfo:0000023 SubClassOf: bfo:0000017
Class: bfo:0000031 SubClassOf: bfo:0000002
Class: bfo:0000040 SubClassOf: bfo:0000004
ObjectProperty: bfo:0000050
ObjectProperty: bfo:0000051
ObjectProperty: bfo:0000054
ObjectProperty: bfo:0000055
ObjectProperty: bfo:0000056
ObjectProperty: bfo:0000057
ObjectProperty: bfo:0000066
"""

PROBE_RE = re.compile(re.escape(TEST_NAMESPACE) + r"#T_[A-Za-z0-9_]+")
LABEL_RE = re.compile(r'"ZZ([A-Za-z0-9_]+)ZZ"')


def humanize(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").strip().lower()


def main() -> int:
    derived = load_catalog(REPO / "src/aegir/ontology/catalog/08_derived.json").templates
    doc, head_iri = RG.render_batch(derived)
    if not head_iri:
        print("no renderable candidates", file=sys.stderr)
        return 1

    # probe IRIs (T_<tid>__<SlotName>, namespaced per-template) -> ONE global sdg: IRI per slot name,
    # so a domain class shared across templates unifies. ZZ markers -> human labels.
    doc = PROBE_RE.sub(lambda m: SDG_NS + m.group(0).split("__")[-1], doc)
    doc = LABEL_RE.sub(lambda m: '"' + humanize(m.group(1)) + '"', doc)
    # inject the numeric-BFO grounding + retitle the ontology
    doc = doc.replace("Ontology: <http://example.org/aegir-batch>",
                      "Ontology: <https://signals360.example.org/sdg>\n" + NUMERIC_BFO)

    ensure_jvm()
    from deeponto.onto import Ontology
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
            f.write(doc)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        r = onto.reasoner.owl_reasoner
        consistent = bool(r.isConsistent())
        n_classes = len(getattr(onto, "owl_classes", {}) or {})
        unsat = []
        if consistent:
            bottom = r.getUnsatisfiableClasses()
            unsat = sorted({str(c.getIRI()) for c in bottom.getEntities().toArray() if not c.isOWLNothing()})
        print(f"REALIZED: consistent={consistent}  named_classes={n_classes}  unsatisfiable={len(unsat)}  "
              f"(from {len(derived)} derived templates)")
        if unsat:
            print("  unsat sample:", [u.split('#')[-1] for u in unsat[:10]])

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "sdg-ontology.omn").write_text(doc)
        owl_ok = False
        try:  # best-effort RDF/XML for owlready2 / Protégé consumers
            import jpype
            fmt = jpype.JClass("org.semanticweb.owlapi.formats.RDFXMLDocumentFormat")()
            jiri = jpype.JClass("org.semanticweb.owlapi.model.IRI")
            jfile = jpype.JClass("java.io.File")
            owl_path = OUT / "sdg-ontology.owl"
            onto.owl_onto.getOWLOntologyManager().saveOntology(onto.owl_onto, fmt, jiri.create(jfile(str(owl_path))))
            owl_ok = owl_path.exists()
        except Exception as e:  # noqa: BLE001
            print(f"  (.owl RDF/XML save skipped: {type(e).__name__}: {str(e)[:80]})")

        cert = (
            "# HermiT consistency certificate — `sdg-ontology`\n\n"
            f"- **isConsistent**: `{consistent}`\n"
            f"- **named classes**: {n_classes}\n"
            f"- **unsatisfiable classes**: {len(unsat)}\n"
            f"- **realized from**: the {len(derived)} FinePDFs-derived templates (`08_derived`)\n"
            "- **reasoner**: HermiT (OWLAPI, via DeepOnto)\n"
            "- **grounding**: BFO 2020 (incl. continuant ⊥ occurrent) + CCO upper\n"
            f"- **generated**: {datetime.date.today().isoformat()} by `scripts/build_realized_ontology.py`\n\n"
            "This is the **realized** ontology — the templates instantiated into concrete OWL axioms, not the\n"
            "`{slot:Type}` skeletons in `catalog/`. Re-verify by loading `sdg-ontology.omn` (or `.owl`) in any\n"
            "OWL reasoner (Protégé/HermiT, ROBOT, owlready2) and checking consistency.\n"
        )
        (OUT / "HERMIT_CERTIFICATE.md").write_text(cert)
        print(f"emitted: sdg-ontology.omn{' + .owl' if owl_ok else ''} + HERMIT_CERTIFICATE.md -> {OUT.relative_to(REPO)}")
        return 0 if (consistent and not unsat) else 2
    finally:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
