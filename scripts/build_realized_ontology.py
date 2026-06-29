#!/usr/bin/env python
"""Realize the FinePDFs-derived templates into a consumer-ready, HermiT-validated BFO/CCO-grounded
domain ontology, and emit it (+ a consistency certificate) into ``corpora/ontology/``.

The catalog ships *templates* (Manchester skeletons with the `{slot:Type}` DSL). This MATERIALIZES the
FinePDFs-derived templates (``08_derived``) into concrete OWL axioms over global ``sdg:`` domain IRIs
(so a class shared by two templates is ONE class), assembles them with the BFO/CCO grounding, applies the
**Phase-A rigor lifts** (filler BFO-grounding, definition annotations, typed DataProperties — realizer-side,
no re-derivation; see EVIDENCE.md OQ-Structure), validates with HermiT, and emits the loadable artifact:

    corpora/ontology/sdg-ontology.omn   the realized ontology (OWL Manchester syntax)
    corpora/ontology/sdg-ontology.owl   the same, RDF/XML (best-effort, for owlready2/Protégé)
    corpora/ontology/HERMIT_CERTIFICATE.md   the consistency certificate

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py
"""
from __future__ import annotations

import argparse
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
# an sdg: domain class as a full IRI (the form render_batch emits after PROBE_RE unification)
SDG_IRI = re.compile(r"<(https://signals360\.example\.org/sdg#[A-Za-z0-9_]+)>")
# name cue: a class whose head noun reads as an occurrent (process/activity) grounds under bfo:0000003
_OCCURRENT_CUE = re.compile(
    r"(Process|Procedure|Activity|Event|Operation|Reaction|Transition|Assessment|Analysis|Behaviou?r|"
    r"Cultivation|Monitoring|Treatment|Processing|Inspection|Verification|Production|Formation|Detection)$")


def humanize(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").strip().lower()


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _head_map(derived) -> dict:
    """{head sdg-IRI → template} — the head is the FIRST ``{Name:Class}`` slot of the manchester_template."""
    out: dict = {}
    for t in derived:
        m = re.search(r"Class:\s*\{(\w+):", t.manchester_template)
        if m:
            out.setdefault(SDG_NS + m.group(1), t)
    return out


_DEGEN = re.compile(r"^Class:\s*<https://signals360\.example\.org/sdg#([A-Z])>(?:\s|$)")


def drop_degenerate(doc: str) -> "tuple[str, list[str]]":
    """Drop stale degenerate template heads — single-letter generic slot names (X/Y/Z), a known
    08_derived staleness artifact. Two templates sharing such a name unify into one class with
    conflicting cross-category anchors (continuant ⊥ occurrent) → unsatisfiable. Phase B's
    re-derivation produces domain-meaningful names and supersedes this filter. Removes the whole
    frame (the ``Class:`` line + its indented continuations)."""
    out: list[str] = []
    dropped: list[str] = []
    skip = False
    for line in doc.split("\n"):
        m = _DEGEN.match(line)
        if m:
            dropped.append(m.group(1))
            skip = True
            continue
        if skip:
            if line.startswith((" ", "\t")):
                continue  # an indented continuation of the dropped frame
            skip = False
        out.append(line)
    return "\n".join(out), sorted(set(dropped))


# ── Phase-A rigor lifts (string post-processing on the assembled OMN, before HermiT) ──

def _filler_anchor(iri: str) -> str:
    """Infer a BFO category for an ungrounded filler from its head noun (occurrent cue → occurrent;
    else continuant). Pure fillers carry no competing parent, so this never collides — the realizer's
    HermiT pass + ``--strict-grounding`` back-off is the safety net regardless."""
    return "bfo:0000003" if _OCCURRENT_CUE.search(iri.split("#")[-1]) else "bfo:0000002"


def ground_fillers(doc: str, exclude: "frozenset[str]" = frozenset()) -> "tuple[str, list[str]]":
    """A.1 — anchor every sdg: class lacking a SubClassOf (a filler) to an inferred BFO category.
    bfo_grounded ~36% → ~95%+. ``exclude`` (full IRIs) skips grounding for classes the --strict-grounding
    back-off found unsatisfiable. Returns (doc, grounded_filler_iris)."""
    declared = set(SDG_IRI.findall(doc))
    # a class is grounded/defined if it carries a SubClassOf OR an EquivalentTo (the genus grounds it)
    grounded = set(re.findall(r"Class:\s*<(https://signals360\.example\.org/sdg#[A-Za-z0-9_]+)>\s+(?:SubClassOf|EquivalentTo):", doc))
    fillers = sorted(declared - grounded - set(exclude))
    if not fillers:
        return doc, []
    block = "\n".join(f"Class: <{iri}> SubClassOf: {_filler_anchor(iri)}" for iri in fillers)
    # NB: no '#' comment lines — the OWLAPI Manchester parser does not treat '#' as a comment and would
    # parse the text as content (manufacturing junk entities + a malformed namespace).
    return doc + "\n\n" + block + "\n", fillers


def annotate_definitions(doc: str, derived) -> "tuple[str, int]":
    """A.2 — emit a definition annotation (iao:0000115 + rdfs:comment) for every sdg: class: the head's
    ``verbal_template`` (a genuine NL definition), or a minimal label gloss for a referenced filler.
    def_annotation_coverage 0% → ~100% (the IAO naturalLanguageDefinition convention)."""
    heads = _head_map(derived)
    declared = sorted(set(SDG_IRI.findall(doc)))
    lines = ["AnnotationProperty: iao:0000115"]
    for iri in declared:
        t = heads.get(iri)
        defn = t.verbal_template if (t and getattr(t, "verbal_template", "")) else f"A {humanize(iri.split('#')[-1])}."
        d = _esc(defn)
        lines.append(f'Class: <{iri}>\n    Annotations: iao:0000115 "{d}", rdfs:comment "{d}"')
    return doc + "\n\n" + "\n".join(lines) + "\n", len(declared)


def emit_datatype_props(doc: str, derived) -> "tuple[str, int]":
    """A.3 — declare + assert typed DataProperties on heads from sdg-vocab.ttl (``ddl._data_properties``),
    keyed by the head's BFO/CCO anchor. AR 0 → >0 (the CPA-relevant attribute inventory)."""
    from aegir.ontology import ddl
    by_dom = ddl._data_properties()
    decls: set = set()
    asserts: list = []
    for iri, t in _head_map(derived).items():
        props: list = []
        for a in (getattr(t, "bfo_anchor_path", []) or []):
            if by_dom.get(a):
                props = by_dom[a]
                break
        for _col, rng, prop_iri in props[:2]:  # 1-2 representative typed attributes per head
            decls.add(f"DataProperty: {prop_iri}")  # Manchester keyword is 'DataProperty:' (not 'DatatypeProperty:')
            asserts.append(f"Class: <{iri}> SubClassOf: {prop_iri} some {rng}")
    if not asserts:
        return doc, 0
    block = "\n".join(sorted(decls)) + "\n" + "\n".join(asserts)
    return doc + "\n\n" + block + "\n", len(asserts)


def _reason(doc: str):
    """Write the OMN to a temp file, load it under HermiT, return (onto, tmp_path, consistent, n_classes, unsat)."""
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(doc)
        path = f.name
    from deeponto.onto import Ontology
    onto = Ontology(path, reasoner_type="hermit")
    r = onto.reasoner.owl_reasoner
    consistent = bool(r.isConsistent())
    n_classes = len(getattr(onto, "owl_classes", {}) or {})
    unsat: list = []
    if consistent:
        bottom = r.getUnsatisfiableClasses()
        unsat = sorted({str(c.getIRI()) for c in bottom.getEntities().toArray() if not c.isOWLNothing()})
    return onto, path, consistent, n_classes, unsat


CCO_TTL = REPO / "build" / "grounding" / "cco-merged.ttl"
CCO_ICE = "cco:ont00000958"  # Information Content Entity (real opaque CCO IRI) — the FHIR-resource bridge target


def import_cco_bridge_fhir(doc: str) -> str:
    """Make CCO a REASONING authority (not just a naming one): align the cco: prefix to CCO's https IRIs, import
    cco-merged.ttl so HermiT validates grounding against CCO's 26 disjointness axioms (it will REJECT
    Plant⊑Vehicle), declare fhir:, and bridge each referenced FHIR type to cco:InformationContentEntity (FHIR
    resources ARE records). bfo: already aligns (purl obo BFO_ in both ontologies), so the chains are coherent."""
    out = doc.replace("Prefix: cco: <http://www.commoncoreontologies.org/>",
                      "Prefix: cco: <https://www.commoncoreontologies.org/>")
    if "Prefix: fhir:" not in out:
        out = out.replace("Prefix: cco: <https://www.commoncoreontologies.org/>\n",
                          "Prefix: cco: <https://www.commoncoreontologies.org/>\nPrefix: fhir: <http://hl7.org/fhir/>\n")
    if CCO_TTL.exists() and "\nImport:" not in out:
        out = re.sub(r"(Ontology: <[^>]+>\n)", rf"\1Import: <file:{CCO_TTL}>\n", out, count=1)
    fhirs = sorted(set(re.findall(r"\bfhir:[A-Za-z][A-Za-z0-9]*", out)))
    if fhirs:
        out = out.rstrip() + "\n\n" + "\n".join(f"Class: {c} SubClassOf: {CCO_ICE}" for c in fhirs) + "\n"
    return out


def consistency_check(templates) -> "tuple[bool, list[str]]":
    """The CCO reasoning-authority membrane for the intermediate-class define loop: render the templates +
    import CCO + run HermiT → (consistent, [unsatisfiable sdg IRIs]). Catches axioms that PARSE but ground an
    intermediate class to a CCO-disjoint or BFO-incompatible genus — so the agent RESPONDS to the reasoner
    (not just the parser). No Phase-A grounding here: we judge the agent's ≡ genus, not the realizer's defaults."""
    doc, head_iri = RG.render_batch(templates)
    if not head_iri:
        return True, []
    doc = PROBE_RE.sub(lambda m: SDG_NS + m.group(0).split("__")[-1], doc)
    doc = LABEL_RE.sub(lambda m: '"' + humanize(m.group(1)) + '"', doc)
    doc = doc.replace("Ontology: <http://example.org/aegir-batch>",
                      "Ontology: <https://signals360.example.org/sdg>\n" + NUMERIC_BFO)
    doc, _ = drop_degenerate(doc)
    doc = import_cco_bridge_fhir(doc)
    ensure_jvm()
    _onto, path, consistent, _n, unsat = _reason(doc)
    Path(path).unlink(missing_ok=True)
    return consistent, unsat


def main() -> int:
    ap = argparse.ArgumentParser(description="realize FinePDFs-derived templates → HermiT-validated OWL")
    ap.add_argument("--no-definitions", action="store_true", help="skip Phase-A.2 definition annotations")
    ap.add_argument("--no-datatype-props", action="store_true", help="skip Phase-A.3 typed DataProperties")
    ap.add_argument("--strict-grounding", action="store_true",
                    help="if Phase-A.1 filler grounding yields an unsatisfiable class, drop it and re-reason")
    args = ap.parse_args()

    derived = load_catalog(REPO / "src/aegir/ontology/catalog/08_derived.json").templates
    doc, head_iri = RG.render_batch(derived)
    if not head_iri:
        print("no renderable candidates", file=sys.stderr)
        return 1

    # probe IRIs (T_<tid>__<SlotName>, namespaced per-template) -> ONE global sdg: IRI per slot name,
    # so a domain class shared across templates unifies. ZZ markers -> human labels. + numeric-BFO grounding.
    doc = PROBE_RE.sub(lambda m: SDG_NS + m.group(0).split("__")[-1], doc)
    doc = LABEL_RE.sub(lambda m: '"' + humanize(m.group(1)) + '"', doc)
    base_doc = doc.replace("Ontology: <http://example.org/aegir-batch>",
                           "Ontology: <https://signals360.example.org/sdg>\n" + NUMERIC_BFO)
    base_doc, degen = drop_degenerate(base_doc)
    base_doc = import_cco_bridge_fhir(base_doc)  # CCO reasoning authority (HermiT validates disjointness) + FHIR bridge
    if degen:
        print(f"   dropped {len(degen)} degenerate stale head(s) (single-letter slots): {degen}")

    def build(ground: bool, exclude: "frozenset[str]" = frozenset()) -> "tuple[str, int, int, int]":
        d, nf, na, nd = base_doc, 0, 0, 0
        if ground:
            d, fillers = ground_fillers(d, exclude=exclude)
            nf = len(fillers)
        if not args.no_definitions:
            d, na = annotate_definitions(d, derived)
        if not args.no_datatype_props:
            d, nd = emit_datatype_props(d, derived)
        return d, nf, na, nd

    ensure_jvm()
    exclude: "set[str]" = set()
    doc, nf, na, nd = build(ground=True)
    onto, path, consistent, n_classes, unsat = _reason(doc)
    # --strict-grounding: greedily drop ONLY the filler-grounding edges that introduce unsatisfiability,
    # re-reasoning until clean (or no further progress) — keeps the bulk of the grounding gain.
    for _ in range(5):
        bad = {u for u in unsat if "signals360" in u} if args.strict_grounding else set()
        if not bad or bad <= exclude:
            break
        exclude |= bad
        print(f"   ⚠ {len(bad)} unsatisfiable — dropping their grounding + re-reasoning ({len(exclude)} excluded)")
        Path(path).unlink(missing_ok=True)
        doc, nf, na, nd = build(ground=True, exclude=frozenset(exclude))
        onto, path, consistent, n_classes, unsat = _reason(doc)
    try:
        print(f"   Phase-A: grounded {nf} fillers · annotated {na} classes · {nd} datatype-prop assertions")
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
            f"- **rigor (Phase A)**: {nf} filler classes BFO-grounded · {na} classes carry NL definitions "
            f"(iao:0000115) · {nd} typed DataProperty assertions\n"
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
