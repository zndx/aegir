#!/usr/bin/env python
"""Realize the FinePDFs-derived templates into a consumer-ready, HermiT-validated BFO/CCO-grounded
domain ontology, and emit it (+ a consistency certificate) into ``corpora/ontology/``.

The catalog ships *templates* (Manchester skeletons with the `{slot:Type}` DSL). This MATERIALIZES the
FinePDFs-derived templates (``catalog.json``) into concrete OWL axioms over global ``sdg:`` domain IRIs
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
import os
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
SDG_NS = "https://signals.zndx.org/sdg#"
SIGNALS_OUT = REPO / "build" / "realize_signals.json"  # boundary signals (unsat justifications) → the re-authoring loop

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
ObjectProperty: bfo:0000054
ObjectProperty: bfo:0000055
ObjectProperty: bfo:0000056
ObjectProperty: bfo:0000057
ObjectProperty: bfo:0000066
ObjectProperty: bfo:0000117
ObjectProperty: bfo:0000132
ObjectProperty: bfo:0000176
ObjectProperty: bfo:0000178
ObjectProperty: bfo:0000196
ObjectProperty: bfo:0000197
"""

# #27 move 1 (STAGED ARMING): AEGIR_RELATION_SIGNATURES=1 appends BFO-faithful Domain/Range/InverseOf
# for the numeric properties + grounds sdg relation coins (SubPropertyOf their nominal BFO parents) —
# turning the ~39 verified miscasts into UNSAT SIGNALS at realize (Sweep-B doctrine). OFF by default
# until move 2 re-authors those axioms; verified isolated by scripts/verify_relation_signatures.py
# (33/33 lint heads + 6 reasoner-only catches).
import os as _os
if _os.environ.get("AEGIR_PROPERTY_DOMAINS", "1") == "1":        # DEFAULT ON (2026-07-22):
    # the arming gate CERTIFIED the armed set (union mass verify: HermiT 0-unsat ·
    # kvasir full-rule no-clash) and the certified frames ship in the sdg-strategy
    # release (methods/armed_property_domains.omn) — the accessor reads by strategy ref
    # and REFUSES on staged-identity mismatch, so this path arms exactly the certified
    # set or fails loudly (the #27 discipline completed at property-domain scale).
    from aegir.ontology.property_domains import armed_domains_omn as _dom
    _armed = _dom()
    # the armed frames may reference bfo:/cco: classes beyond the catalog scaffold —
    # Manchester requires declarations at use, so declare every referenced class not
    # already present (the first full realize under the flip found bfo:0000016 undeclared)
    import re as _re3
    _refs = set(_re3.findall(r"(?:Domain|Range):\s*((?:bfo|cco):[\w]+)", _armed))
    _missing = sorted(c for c in _refs if f"Class: {c}" not in NUMERIC_BFO + _armed)
    _decls = "".join(f"\nClass: {c}\n" for c in _missing)
    NUMERIC_BFO = NUMERIC_BFO + _decls + _armed   # exclusion-aware: demoted ∪ deferred filtered
# The relational-concepts closure (RH 2026-07-23): the ontology extends over its own
# relational projection — pattern + column-role classes, authored (not FinePDFs-derived),
# ⊑ cco ICE, shipped in every release so a database-only consumer can tag every entity.
from aegir.ontology.relational_concepts import RELATIONAL_CONCEPTS_OMN as _RELC
NUMERIC_BFO = NUMERIC_BFO + _RELC
# spec interop — PRODML + SysMLv2 mapped entities (RH 2026-07-23: certain, RC-released;
# rdfs:seeAlso carries the source-spec join for HDF5/database-side consumers).
from aegir.ontology.spec_mappings import SPEC_MAPPINGS_OMN as _SPEC
NUMERIC_BFO = NUMERIC_BFO + _SPEC
if _os.environ.get("AEGIR_RELATION_SIGNATURES", "1") != "0":     # DEFAULT ON since 2026-07-19:
    # the 39 miscasts are re-authored (UNSAT=0 verified) — signatures now guard every realize.
    from aegir.ontology.relation_signatures import SIGNATURES_OMN as _SIG
    NUMERIC_BFO = NUMERIC_BFO + _SIG

PROBE_RE = re.compile(re.escape(TEST_NAMESPACE) + r"#T_[A-Za-z0-9_]+")
LABEL_RE = re.compile(r'"ZZ([A-Za-z0-9_]+)ZZ"')
# an sdg: domain class as a full IRI (the form render_batch emits after PROBE_RE unification)
SDG_IRI = re.compile(r"<(https://signals\.zndx\.org/sdg#[A-Za-z0-9_]+)>")
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


_DEGEN = re.compile(r"^Class:\s*<https://signals\.zndx\.org/sdg#([A-Z])>(?:\s|$)")


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
    grounded = set(re.findall(r"Class:\s*<(https://signals\.zndx\.org/sdg#[A-Za-z0-9_]+)>\s+(?:SubClassOf|EquivalentTo):", doc))
    fillers = sorted(declared - grounded - set(exclude))
    if not fillers:
        return doc, []
    block = "\n".join(f"Class: <{iri}> SubClassOf: {_filler_anchor(iri)}" for iri in fillers)
    # NB: no '#' comment lines — the OWLAPI Manchester parser does not treat '#' as a comment and would
    # parse the text as content (manufacturing junk entities + a malformed namespace).
    return doc + "\n\n" + block + "\n", fillers


_BACKBONE_PARENTS = None


def _backbone_parents() -> dict:
    """CCO's subClassOf backbone (cached), keyed by full IRI — real cco: *opaque* IRIs → … → BFO. Mirrors the
    metrology's own merge so grounding-reachability agrees with the metrology's bfo_grounded."""
    global _BACKBONE_PARENTS
    if _BACKBONE_PARENTS is None:
        import rdflib
        from rdflib import RDFS, URIRef
        _BACKBONE_PARENTS = {}
        cco = REPO / "build" / "grounding" / "cco-merged.ttl"
        if cco.exists():
            for s, _, o in rdflib.Graph().parse(str(cco), format="turtle").triples((None, RDFS.subClassOf, None)):
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    _BACKBONE_PARENTS.setdefault(str(s), []).append(str(o))
    return _BACKBONE_PARENTS


def ground_closure(doc: str) -> "tuple[str, int]":
    """A.1b — ground every sdg: class that cannot actually REACH BFO. ``ground_fillers`` trusts any
    SubClassOf/≡ as grounding, but the engine's ≡ genera include CCO refs it invented as readable http IRIs
    (``cco:ont00000853``) that don't match CCO's real opaque https IRIs — so they never resolve to BFO —
    and sdg: genus chains can dead-end. Compute real reachability (subClassOf + ≡-first-genus, prefixes
    resolved, CCO backbone merged) and add a DIRECT ``SubClassOf: <bfo category>`` for each unreachable sdg:
    class — a correct coarse grounding that leaves the ≡ intact. Recovers bfo_grounded to what the ontology
    actually warrants (the metrology's anchor-walk agrees)."""
    pfx = dict(re.findall(r"Prefix:\s*(\w+):\s*<([^>]+)>", doc))
    bfo_ns = [x for x in (pfx.get("bfo"), "http://purl.obolibrary.org/obo/BFO_") if x]

    def resolve(tok: str) -> str:
        tok = tok.strip()
        if tok.startswith("<") and tok.endswith(">"):
            return tok[1:-1]
        if ":" in tok and tok.split(":", 1)[0] in pfx:
            p, local = tok.split(":", 1)
            return pfx[p] + local
        return tok

    parents = {k: list(v) for k, v in _backbone_parents().items()}
    doc_refs: set = set()
    for m in re.finditer(r"Class:\s*(<[^>]+>|\w+:[\w-]+)\s+(?:SubClassOf|EquivalentTo):\s*([^\n]+)", doc):
        x = resolve(m.group(1))
        doc_refs.add(x)
        gm = re.match(r"\s*(<[^>]+>|\w+:[\w-]+)", m.group(2))
        if gm:
            g = resolve(gm.group(1))
            parents.setdefault(x, []).append(g)
            doc_refs.add(g)

    # Ground the TERMINAL dead-ends the ≡ chains bottom out at — the parentless, non-BFO IRIs the doc references
    # (the fictional cco: genera the engine invented + sdg: leaves ground_fillers didn't reach). Grounding THESE
    # (not the ≡-heads) lets each head reach BFO THROUGH its genus with no 2nd parent, so bfo_grounded recovers
    # with NO tangledness penalty (a head ≡ sdg:Y ⊓ …, Y ≡ sdg:Fictional resolves once sdg:Fictional is grounded).
    ung = sorted(c for c in doc_refs
                 if c.startswith("http") and not parents.get(c) and not any(c.startswith(b) for b in bfo_ns))
    if not ung:
        return doc, 0
    block = "\n".join(f"Class: <{c}> SubClassOf: {_filler_anchor(c)}" for c in ung)
    return doc + "\n\n" + block + "\n", len(ung)


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


def drop_classes(doc: str, iris: "list[str]") -> str:
    """REMOVE every targeted Class frame entirely — for INTERNAL probe classes (ABox conjunction probes,
    kvasir pre-screen) that must NEVER appear in the published artifact or its grounding count. For an
    unsatisfiable REAL class use degrade_classes (keep a bare declaration so references still resolve)."""
    return _shed_classes(doc, iris, keep_declaration=False)


# a Class head appears in EITHER form in the rendered doc — full IRI `Class: <…#Name>` OR prefixed
# `Class: sdg:Name`; match both (an IRI-only pattern silently misses every prefixed head).
_CLASS_HEAD_RE = re.compile(r"^Class:\s+(<[^>]*#([A-Za-z0-9_]+)>|sdg:([A-Za-z0-9_]+))")


def _shed_classes(doc: str, iris: "list[str]", keep_declaration: bool) -> str:
    """Shed the body of every targeted Class frame. ``keep_declaration`` False REMOVES the frame outright
    (for INTERNAL probe classes that must not reach the published artifact); True keeps a bare declaration."""
    targets = {i.split("#")[-1] for i in iris}
    out, skipping = [], False
    for line in doc.split("\n"):
        m = _CLASS_HEAD_RE.match(line)
        if m:
            if (m.group(2) or m.group(3)) in targets:
                if keep_declaration:
                    out.append(f"Class: {m.group(1)}")  # keep the declaration (same form); shed the unsat body
                skipping = True
                continue
            skipping = False
        elif line[:1].strip():  # a new top-level frame ends the shed
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def degrade_classes(doc: str, iris: "list[str]") -> str:
    """DEGRADE every targeted Class frame to a BARE declaration — for an unsatisfiable REAL class: keep it
    DECLARED (references resolve, the document parses) while shedding the unsat body; the full axiom is
    recovered by reauthor_unsat from the emitted signal (task #14). Contrast drop_classes, which REMOVES the
    frame (for internal probe classes that must never appear in the published artifact)."""
    return _shed_classes(doc, iris, keep_declaration=True)


# Default 90 min (RH 2026-07-02: 30 min to start, "~90 minutes acceptable" — real, meaningful ontologies
# take time to validate; validation TIME is a cost we pay willingly, ontology SUBSTANCE is not). The wall
# exists to catch the PATHOLOGICAL grind (the ABox×≡ mode), never to rush the reasoner — and when it fires,
# the response ORDER is: pay more time first; shape the INPUT only with a stated losslessness argument
# (e.g. a ⊥-module preserves all entailments over its signature); NEVER thin the curated ontology itself —
# purposeful, substantive curation is a core objective (it grounds the DDL+views corpus that drives the
# mixed text+views pre-training corpus).
REASON_BUDGET_S = int(os.environ.get("AEGIR_REASON_BUDGET_S", "5400"))


def _reason(doc: str, explain: bool = False):
    """Write the OMN to a temp file, load it under HermiT, return (onto, tmp_path, consistent, n_classes, unsat, why).

    This is the realize BOUNDARY — the one place a full-context logical conflict (a class the derivation
    authored that is empty against BFO + π(CCO)) first becomes visible. On its own it emits only a NAME
    ("X is unsatisfiable"), a signal no agent can adapt to. With ``explain=True`` it also emits the SIGNAL:
    ``why[iri] = {"axioms": [minimal justification], "why": <legible cause>}`` (via aegir.ontology.explain),
    so the offending conjunct can be RE-AUTHORED, not just shed. Off by default because justification search
    is expensive; only the actionable drop/halt sets need it, and when ``unsat`` is empty ``why`` is ``{}``
    at zero cost."""
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(doc)
        path = f.name
    # WALL-CLOCK BUDGET (RH 2026-07-02: HermiT must not fall into a pathological grind on trivial-but-
    # expensive inputs — measured: 248 Types-only individuals × 374 ≡-classes ground a single core >20 min
    # unfinished vs ~3 min TBox-only). A JVM tableau can't be interrupted in-process, so the watchdog is a
    # process wall: past AEGIR_REASON_BUDGET_S it emits the tractability SIGNAL and exits 3 (distinct from
    # 2=inconsistent). The reasoner is ground truth; the budget is how long we let it testify.
    import threading
    import time as _time
    t0 = _time.monotonic()
    done = threading.Event()

    def _watchdog() -> None:
        if not done.wait(REASON_BUDGET_S):
            print(f"✘ REASONING BUDGET EXCEEDED ({REASON_BUDGET_S}s; AEGIR_REASON_BUDGET_S) — a pathological "
                  "input is grinding HermiT (ABox × ≡-classes is the known shape; retry with "
                  "--no-individuals, or raise the budget deliberately)", file=sys.stderr, flush=True)
            os._exit(3)

    threading.Thread(target=_watchdog, daemon=True).start()
    from deeponto.onto import Ontology
    onto = Ontology(path, reasoner_type="hermit")
    r = onto.reasoner.owl_reasoner
    consistent = bool(r.isConsistent())
    n_classes = int(onto.owl_onto.getClassesInSignature().size())
    n_inds_sig = int(onto.owl_onto.getIndividualsInSignature().size())
    unsat: list = []
    if consistent:
        bottom = r.getUnsatisfiableClasses()
        unsat = sorted({str(c.getIRI()) for c in bottom.getEntities().toArray() if not c.isOWLNothing()})
    why: dict = {}
    if explain and unsat:
        from aegir.ontology.explain import explain_unsatisfiable
        why = explain_unsatisfiable(onto.owl_onto, r, unsat)
    done.set()
    print(f"   reasoned in {_time.monotonic() - t0:.0f}s (classes={n_classes} individuals={n_inds_sig} "
          f"consistent={consistent} unsat={len(unsat)}; budget {REASON_BUDGET_S}s)", flush=True)
    return onto, path, consistent, n_classes, unsat, why


def _write_signals(signals: "dict[str, dict]") -> None:
    """Surface the boundary's per-class justifications to a stable record — the interface between the
    reasoner boundary and the agent that re-authors the offending conjunct. Each entry is
    ``{iri: {"axioms": [...], "why": ...}}`` for a class the realize narrowed out of the domain."""
    import json
    SIGNALS_OUT.parent.mkdir(parents=True, exist_ok=True)
    SIGNALS_OUT.write_text(json.dumps(signals, indent=2, sort_keys=True))
    print(f"   ⚑ {len(signals)} boundary signal(s) → {SIGNALS_OUT.relative_to(REPO)} (for re-authoring)")


CCO_TTL = REPO / "build" / "grounding" / "cco-module.ttl"  # π(CCO): the ⊥-locality module — BFO + π is the one fixed theory (gate-certified tractable; full cco-merged.ttl is intractable, see build_cco_module.py)
CCO_ICE = "cco:ont00000958"  # Information Content Entity (real opaque CCO IRI) — the FHIR-resource bridge target

# RETIRED 2026-07-11 (RH, with extreme prejudice): CCO_READABLE_ALIASES was a hardcoded dict that rewrote
# fictional readable cco: genera (cco:ont00000965) to real CCO IRIs at realize. That was an ANTI-PATTERN —
# it silently BLESSED the hallucination (reinforcing DirectiveICE across refinement) and masked a broader  coined-ok
# contamination it never covered (cco:BusinessEntity, cco:has_part, legacy bfo:0000052 — coined-ok: names to reject). External namespaces
# are AUTHORITIES, not sandboxes: the fix is the HARD verification gate (aegir.ontology.external_index,
# wired into evolve_rigor.validate_detailed) that REJECTS any cco:/bfo: reference not existing in the current
# authoritative ontology, forcing the agent to use a real IRI or coin sdg:. Existing contamination was
# resolved once through that index (scripts/resolve_contamination.py). See [[bfo_cco_grounding_mandate]].


def import_cco_bridge_fhir(doc: str) -> str:
    """Align the cco: prefix to CCO's real https IRIs (so our 140 cco:ont refs resolve to CCO's deep hierarchy  # coined-ok: prose/placeholder, not a real ref
    for grounding), declare fhir:, bridge each referenced FHIR type to cco:ont00000958, and import
    π(CCO) — the ⊥-locality module (CCO_TTL) — to make CCO a REASONING authority: HermiT validates grounding
    against CCO's disjointness (it REJECTs Plant⊑Vehicle). π(CCO) replaces full CCO, which is intractable once
    hundreds of ≡ interact with its inverse/transitive/⊔; the module is lossless over our signature, so every
    disjointness that could refute a grounding is preserved. There is no knob to skip it: BFO + π(CCO) is THE
    theory, always applied. bfo: already aligns (purl obo BFO_ in both), so the chains are coherent."""
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
                      "Ontology: <https://signals.zndx.org/sdg>\n" + NUMERIC_BFO)
    doc, _ = drop_degenerate(doc)
    doc = import_cco_bridge_fhir(doc)  # always aligns cco: NS; self-gates the CCO import
    doc, _ = RG.declare_used_properties(doc)
    # kvasir fast-refute pre-pass (P0, [[greenfield_reasoner_direction]]): a kernel-checked
    # refutation short-circuits the HermiT round in milliseconds, and its minimal justification IS
    # the reason the agent re-authors against. Any other verdict (no-clash / out-of-fragment /
    # unavailable) falls through to HermiT unchanged — kvasir changes HOW FAST, never WHAT is
    # verified; HermiT keeps sole certificate authority. Each no-clash-then-HermiT pair accretes
    # to the differential record (trust is measured, never assumed).
    fr = None
    try:
        from aegir.ontology.kvasir_bridge import differential_record, fast_refute
        fr = fast_refute(doc, source="consistency_check")
        if fr["verdict"] == "refuted":
            k_unsat = [u for u in fr.get("unsat_classes", []) if SDG_NS in u]
            if k_unsat:
                print(f"  ⚡ kvasir refuted in {fr['ms']}ms — {len(k_unsat)} unsat, HermiT round "
                      f"short-circuited: {fr.get('reason', '')[:200]}", flush=True)
                return False, k_unsat
    except Exception as e:  # noqa: BLE001 — the pre-pass must never block the oracle
        print(f"  (kvasir pre-pass unavailable: {e})", file=sys.stderr)
    ensure_jvm()
    _onto, path, consistent, _n, unsat, _why = _reason(doc)
    Path(path).unlink(missing_ok=True)
    if fr is not None and fr.get("verdict") in ("no-clash", "refuted"):
        try:
            differential_record("consistency_check", fr["verdict"], consistent,
                                n_axioms=fr.get("n_axioms"), kvasir_ms=fr.get("ms"))
        except Exception:  # noqa: BLE001
            pass
    return consistent, unsat


def main() -> int:
    ap = argparse.ArgumentParser(description="realize FinePDFs-derived templates → HermiT-validated OWL")
    ap.add_argument("--no-definitions", action="store_true", help="skip Phase-A.2 definition annotations")
    ap.add_argument("--no-datatype-props", action="store_true", help="skip Phase-A.3 typed DataProperties")
    ap.add_argument("--strict-grounding", action="store_true",
                    help="if Phase-A.1 filler grounding yields an unsatisfiable class, drop it and re-reason")
    ap.add_argument("--no-individuals", action="store_true",
                    help="realize the TBox only (skip the individual registry's ABox)")
    args = ap.parse_args()
    if not CCO_TTL.exists():  # INVARIANT: the theory must be present — never realize against a vacuous CCO-less check
        print(f"✘ the theory π(CCO) is not built ({CCO_TTL.name}) — run scripts/build_cco_module.py first; "
              "refusing to realize against a vacuous, CCO-less theory", file=sys.stderr)
        return 2

    derived = load_catalog(REPO / "src/aegir/ontology/catalog/catalog.json").templates
    doc, head_iri = RG.render_batch(derived)
    if not head_iri:
        print("no renderable candidates", file=sys.stderr)
        return 1

    # probe IRIs (T_<tid>__<SlotName>, namespaced per-template) -> ONE global sdg: IRI per slot name,
    # so a domain class shared across templates unifies. ZZ markers -> human labels. + numeric-BFO grounding.
    doc = PROBE_RE.sub(lambda m: SDG_NS + m.group(0).split("__")[-1], doc)
    doc = LABEL_RE.sub(lambda m: '"' + humanize(m.group(1)) + '"', doc)
    base_doc = doc.replace("Ontology: <http://example.org/aegir-batch>",
                           "Ontology: <https://signals.zndx.org/sdg>\n" + NUMERIC_BFO)
    base_doc, degen = drop_degenerate(base_doc)
    base_doc = import_cco_bridge_fhir(base_doc)  # aligns cco: NS + imports π(CCO) — BFO + π is the one fixed theory
    if degen:
        print(f"   dropped {len(degen)} degenerate stale head(s) (single-letter slots): {degen}")

    # Sibling adjudications (RH 2026-07-09): adjudicated-disjoint pairs land as Manchester
    # DisjointClasses frames INSIDE the HermiT boundary — every realize re-validates every
    # adjudication against the full theory + ABox (a co-typed individual refutes it loudly).
    # kvasir parses these natively (Manchester-only tool). Overlap verdicts are record-only.
    from aegir.ontology import adjudication as ADJ
    _adj_frames = ADJ.disjoint_manchester_frames(ADJ.load_adjudications())
    if _adj_frames:
        # NB (line 140): NO '#' comment lines — OWLAPI's Manchester parser does not treat '#' as a comment
        # and the whole document fails to load (the OBO/KRSS2/RDFa cascade all reject it). Append frames only.
        base_doc = base_doc.rstrip() + "\n\n" + "\n".join(_adj_frames) + "\n"
        print(f"   sibling adjudications: {len(_adj_frames)} DisjointClasses frames appended")

    # ABox: the individual registry's membrane-admitted, class-typed individuals — the ontology
    # INSTANTIATED (Convert 1b). Held ASIDE here and joined only after the TBox converges: the ABox pass
    # costs > the whole TBox pass (ladder 2026-07-02: TBox 480s; +248 individuals > 1200s — instance
    # classification ≈ individuals × ≡-classes), so the strict-grounding/narrowing rounds must not re-pay
    # it. "Least-trusted layer" also means ENTERS LAST — one certification pass against the clean theory.
    abox_block, n_inds = "", 0
    probe_sets: "dict[str, frozenset]" = {}
    kvasir_probe_unsat: "set[str]" = set()
    if not args.no_individuals:
        from aegir.ontology import individuals as IND
        _reg = IND.load_registry()
        abox_block = IND.abox_manchester(_reg)
        if abox_block:
            n_inds = IND.n_individuals(_reg)
            # CONJUNCTION PROBES (the decomposition, folded into the main pass): each distinct multi-typed
            # conjunction becomes a probe class judged by getUnsatisfiableClasses alongside everything else
            # — HermiT's internal batching/caching prices these at ~0.5s/class, vs 13-20s+ per FRESH
            # isSatisfiable expression call (measured 2026-07-02: 166 fresh calls ground >57 min, outside
            # any budget/print — the uninstrumented phase this design deletes). Probes ride the budgeted,
            # timed _reason; they are instrumentation, stripped before emission.
            tbi = IND.types_by_individual(_reg)
            for k, ts in enumerate(sorted({frozenset(ts) for ts in tbi.values() if len(ts) > 1},
                                          key=sorted)):
                probe_sets[f"__conjprobe_{k}"] = ts
            if probe_sets:
                probes = "\n".join(
                    f"Class: <{SDG_NS}{name}> EquivalentTo: "
                    + " and ".join(f"<{SDG_NS}{c}>" for c in sorted(ts))
                    for name, ts in probe_sets.items())
                base_doc = base_doc.rstrip() + "\n\n" + probes + "\n"
            print(f"   instantiation queued: {n_inds} individuals · {len(probe_sets)} conjunction probes "
                  f"folded into the main pass (multi-typed sets; singletons ride unsat=∅); "
                  f"budget {REASON_BUDGET_S}s/pass", flush=True)
            # kvasir probe PRE-SCREEN (P0, [[greenfield_reasoner_direction]]): refute what is
            # refutable in milliseconds BEFORE the budgeted HermiT pass — pre-refuted conjunctions
            # withhold through the SAME signal path (kvasir refutations are kernel-checked and
            # definitive), their probe classes leave the doc (a smaller pass), and the operator
            # sees the clash census at second 0 instead of minute 40. HermiT remains the oracle
            # for everything kvasir cannot decide.
            try:
                from aegir.ontology.kvasir_bridge import fast_refute as _kv_refute
                _fr = _kv_refute(base_doc, source="probe-prescreen")
                if _fr["verdict"] == "refuted":
                    kvasir_probe_unsat = {u.rsplit("#", 1)[-1] for u in _fr.get("unsat_classes", [])
                                          if "__conjprobe_" in u}
                    if kvasir_probe_unsat:
                        base_doc = drop_classes(base_doc, [SDG_NS + n for n in kvasir_probe_unsat])
                        print(f"   ⚡ kvasir pre-screen ({_fr['ms']}ms): {len(kvasir_probe_unsat)} "
                              f"conjunction probe(s) pre-refuted → withheld via the standard path, "
                              f"dropped from the HermiT doc; e.g. {_fr.get('reason', '')[:160]}",
                              flush=True)
                    _dom = [u.split('#')[-1] for u in _fr.get("unsat_classes", [])
                            if "__conjprobe_" not in u and SDG_NS in u]
                    if _dom:
                        print(f"   ⚡ kvasir pre-screen also refutes {len(_dom)} domain class(es) — "
                              f"the narrowing loop confirms via HermiT: {_dom[:5]}", flush=True)
                else:
                    print(f"   ⚡ kvasir pre-screen ({_fr['ms']}ms): "
                          f"{_fr['verdict']} over {_fr.get('n_axioms', '?')} lowered axioms", flush=True)
                # ∃-cycle lint — the tableau-grind early-warning (the JIE incident's shape),
                # surfaced BEFORE the pass instead of discovered by a jstack 40 minutes in
                from aegir.ontology.kvasir_bridge import exist_cycle_lint, lower_manchester
                _cycles = exist_cycle_lint(lower_manchester(base_doc)[0])
                if _cycles:
                    print(f"   ⚠ ∃-cycle lint: {len(_cycles)} existential cycle(s) — budget the pass "
                          f"accordingly (largest: {[c.split('#')[-1] for c in _cycles[0][:5]]})",
                          flush=True)
            except Exception as _e:  # noqa: BLE001 — the pre-screen must never block the oracle
                print(f"   (kvasir pre-screen unavailable: {_e})", file=sys.stderr)

    def build(ground: bool, exclude: "frozenset[str]" = frozenset()) -> "tuple[str, int, int, int]":
        d, nf, na, nd = base_doc, 0, 0, 0
        if ground:
            d, fillers = ground_fillers(d, exclude=exclude)
            nf = len(fillers)
        if not args.no_definitions:
            d, na = annotate_definitions(d, derived)
        if not args.no_datatype_props:
            d, nd = emit_datatype_props(d, derived)
        d, _undecl = RG.declare_used_properties(d)
        if _undecl:
            print(f"   auto-declared {len(_undecl)} used-but-undeclared propert{'y' if len(_undecl)==1 else 'ies'}: {_undecl}")
        d, n_closure = ground_closure(d)
        if n_closure:
            print(f"   grounding-closure: +{n_closure} direct BFO anchors (sdg: classes whose ≡ genera don't reach BFO)")
        return d, nf, na, nd

    def _degraded(nc: int, d: str) -> bool:
        # NON-VACUITY floor: a Manchester parse that silently falls back leaves almost nothing in the
        # signature while isConsistent() stays trivially True — the masked-inconsistency failure mode.
        # The signature must carry at least half the doc's DISTINCT declared class IRIs (frames ≠ classes:
        # a class spans several `Class: <iri>` frames — declaration/annotations/≡ — so counting frame
        # occurrences false-positived at 830 real classes vs 2182 frames).
        declared = len(set(re.findall(r"\nClass: (<[^>]+>)", d)))
        return nc < max(50, declared // 2)

    ensure_jvm()
    exclude: "set[str]" = set()
    doc, nf, na, nd = build(ground=True)
    onto, path, consistent, n_classes, unsat, why = _reason(doc, explain=True)
    if _degraded(n_classes, doc):  # INVARIANT: never emit a vacuously-"consistent" artifact
        print(f"✘ parse degraded ({n_classes} classes in signature vs {doc.count(chr(10) + 'Class: <')} "
              "declared) — a fallback parser swallowed the document; refusing to emit a vacuous artifact",
              file=sys.stderr)
        return 2
    # conjunction PROBES are instrumentation, not content: partition their verdicts out of every loop —
    # an unsat probe is an ABox withhold decision, never a grounding back-off or narrowing target.
    def _split_probes(us: "list") -> "tuple[list, set]":
        real = [u for u in us if "__conjprobe_" not in u]
        probes = {u.rsplit("#", 1)[-1] for u in us if "__conjprobe_" in u}
        return real, probes

    probe_unsat: "set[str]" = set(kvasir_probe_unsat)  # kvasir pre-refuted conjunctions withhold too
    unsat, pu = _split_probes(unsat)
    probe_unsat |= pu
    # --strict-grounding: greedily drop ONLY the filler-grounding edges that introduce unsatisfiability,
    # re-reasoning until clean (or no further progress) — keeps the bulk of the grounding gain.
    for _ in range(5):
        bad = {u for u in unsat if SDG_NS in u} if args.strict_grounding else set()
        if not bad or bad <= exclude:
            break
        exclude |= bad
        print(f"   ⚠ {len(bad)} unsatisfiable — dropping their grounding + re-reasoning ({len(exclude)} excluded)")
        Path(path).unlink(missing_ok=True)
        doc, nf, na, nd = build(ground=True, exclude=frozenset(exclude))
        onto, path, consistent, n_classes, unsat, why = _reason(doc, explain=True)
        unsat, pu = _split_probes(unsat)
        probe_unsat |= pu
    # Domain-narrowing is the boundary where the conflict is finally VISIBLE: a class still unsatisfiable
    # after the grounding back-off carries a bad DERIVED axiom (a mis-used BFO role / a filler grounded
    # into a disjoint category). Emit the SIGNAL — the minimal justification per class (why + axioms),
    # recorded for the re-authoring loop — BEFORE dropping, so a shed class is something the agent can
    # re-author, not a silent last-resort bailout.
    signals: dict = {}
    for _ in range(8):
        if not unsat:
            break
        signals.update({k: v for k, v in why.items() if "__conjprobe_" not in k})
        for iri in unsat:
            sig = why.get(iri, {})
            print(f"   ⚠ narrowing: {iri.split('#')[-1]} — {sig.get('why', 'unsatisfiable vs the theory')}")
            for ax in sig.get("axioms", [])[:5]:
                print(f"       · {ax}")
        doc = degrade_classes(doc, unsat)  # keep unsat REAL classes as bare declarations (references resolve)
        Path(path).unlink(missing_ok=True)
        onto, path, consistent, n_classes, unsat, why = _reason(doc, explain=True)
        unsat, pu = _split_probes(unsat)
        probe_unsat |= pu
    if unsat:
        signals.update({k: v for k, v in why.items() if "__conjprobe_" not in k})
    if signals:
        _write_signals(signals)  # the surfaced record — the interface to the re-authoring loop
    elif SIGNALS_OUT.exists():
        SIGNALS_OUT.unlink()  # clean build: clear any stale signal record
    if unsat:  # INVARIANT: the emitted artifact MUST be consistent w.r.t. BFO+π(CCO) — halt, never emit inconsistent
        print(f"✘ {len(unsat)} class(es) remain unsatisfiable after domain-narrowing — cannot emit a consistent "
              f"artifact against the theory (fix the generation):", file=sys.stderr)
        for iri in unsat[:12]:
            print(f"    {iri.split('#')[-1]}: {why.get(iri, {}).get('why', '?')}", file=sys.stderr)
        return 2

    # ── DECOMPOSED ABox certification (the greenfield move, RH 2026-07-02: own the decomposition calculus,
    # keep HermiT as the atomic oracle). THEOREM (our shape): with a NOMINAL-FREE TBox (gate-verified) and a
    # Types-only ABox (no Facts/SameAs — enforced by abox_manchester's construction), the KB is consistent
    # ⟺ the TBox is consistent ∧ every asserted type-conjunction is satisfiable. Singleton type-sets ride
    # unsat=∅ above; the multi-typed conjunctions were judged as PROBE CLASSES inside the main budgeted,
    # timed pass (HermiT's batched per-class tests ≈ 0.5s/class, vs 13-20s+ per fresh isSatisfiable
    # expression call — the uninstrumented, unbudgeted loop this design replaced). Here we only read the
    # verdicts, withhold clashing conjunctions' individuals, and STRIP the probes from the artifact.
    n_withheld = 0
    if abox_block:
        from aegir.ontology import individuals as IND2
        # regenerate the ABox against the FINAL doc's declared classes (the writer-side coherence gate):
        # seeding ran against an earlier catalog state, so entity_classes may reference classes the
        # converged TBox no longer declares (withheld conjuncts, degenerate slots) — one ghost Types:
        # name fails the whole Manchester parse downstream.
        declared = set(re.findall(rf"Class: <{re.escape(SDG_NS)}([A-Za-z0-9_]+)>", doc))
        abox_block = IND2.abox_manchester(IND2.load_registry(), declared=declared)
        n_ghosted = n_inds - abox_block.count("Individual:")
        if n_ghosted:
            print(f"   ⚠ {n_ghosted} individual(s) skipped — typed only by classes the converged TBox "
                  "does not declare (ghost types; re-seed reconciles)", flush=True)
            n_inds -= n_ghosted
        tbi = IND2.types_by_individual(IND2.load_registry())
        bad_sets = [probe_sets[p] for p in sorted(probe_unsat) if p in probe_sets]
        if bad_sets:
            bad_lookup = set(bad_sets)
            withheld_ids = {i for i, ts in tbi.items() if frozenset(ts) in bad_lookup}
            n_withheld = len(withheld_ids)
            for ts in bad_sets:  # the boundary SIGNAL: these type-conjunctions are disjoint under the theory
                print(f"   ⚠ instance-level clash: {{{', '.join(sorted(ts))}}} is UNSATISFIABLE — "
                      f"withholding its individuals (re-author in individual_registry.json)", file=sys.stderr)
            # persist the instance-level signals (the re-seed/re-author interface — prints are not a record)
            import json as _json
            abox_signals = [{"conjunction": sorted(ts),
                             "individuals": sorted(i for i, t2 in tbi.items() if frozenset(t2) == ts)}
                            for ts in bad_sets]
            (REPO / "build" / "abox_clash_signals.json").write_text(_json.dumps(abox_signals, indent=1))
            print(f"   ⚑ {len(bad_sets)} instance-level clash signal(s) → build/abox_clash_signals.json",
                  flush=True)
            abox_block = "\n\n".join(f for f in abox_block.split("\n\n")
                                     if not any(f"<{SDG_NS}{i}>" in f for i in withheld_ids))
            n_inds -= n_withheld
        if probe_sets:  # strip the instrumentation from the artifact
            doc = drop_classes(doc, [SDG_NS + p for p in probe_sets])
        if abox_block:
            doc = doc.rstrip() + "\n\n" + abox_block + "\n"
        print(f"   ABox certified by decomposition: {n_inds} individuals ({len(probe_sets)} conjunction "
              f"probes judged in the main pass, {len(bad_sets)} clashes withheld)", flush=True)

    try:
        print(f"   Phase-A: grounded {nf} fillers · annotated {na} classes · {nd} datatype-prop assertions")
        print(f"REALIZED: consistent={consistent}  named_classes={n_classes}  unsatisfiable={len(unsat)}  "
              f"(from {len(derived)} derived templates)")
        if unsat:
            print("  unsat sample:", [u.split('#')[-1] for u in unsat[:10]])

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "sdg-ontology.omn").write_text(doc)
        owl_ok = False
        artifact_classes = n_classes - len(probe_sets)  # the reason pass counted the (stripped) probes
        try:  # best-effort RDF/XML for owlready2 / Protégé consumers — re-parse the FINAL doc via the
            # PROVEN DeepOnto loader (raw OWLManager auto-detection parse-degraded the Manchester doc on
            # first live use, 2026-07-02 — the guard caught it). Guards: classes AND individuals present.
            import jpype
            from deeponto.onto import Ontology as _Onto
            o2 = _Onto(str(OUT / "sdg-ontology.omn"), reasoner_type="hermit").owl_onto
            nc2 = int(o2.getClassesInSignature().size())
            ni2 = int(o2.getIndividualsInSignature().size())
            if _degraded(nc2, doc) or ni2 < n_inds:
                raise RuntimeError(f"final-doc parse degraded (classes={nc2} individuals={ni2}/{n_inds})")
            artifact_classes = nc2
            fmt = jpype.JClass("org.semanticweb.owlapi.formats.RDFXMLDocumentFormat")()
            jiri = jpype.JClass("org.semanticweb.owlapi.model.IRI")
            jfile = jpype.JClass("java.io.File")
            owl_path = OUT / "sdg-ontology.owl"
            o2.getOWLOntologyManager().saveOntology(o2, fmt, jiri.create(jfile(str(owl_path))))
            owl_ok = owl_path.exists()
        except Exception as e:  # noqa: BLE001
            print(f"  (.owl RDF/XML save skipped: {type(e).__name__}: {str(e)[:80]})")

        # LOGICAL grounding certificate — reasoner-ENTAILED BFO/CCO subsumption per class (the metric
        # reads THIS, not the syntactic rdflib edge-walk, so nth-order/earned grounding is credited;
        # RH 2026-07-11, [[bfo_cco_grounding_mandate]]). TBox-only (~3s). Pre-kvasir HermiT; kvasir-destined.
        try:
            from aegir.ontology.grounding import write_certificate
            _gc = write_certificate(OUT / "sdg-ontology.omn", OUT / "grounding_certificate.json")
            print(f"   grounding certificate: {_gc['grounded_count']}/{_gc['n']} = {_gc['rate']} "
                  f"BFO/CCO-grounded (HermiT-entailed)")
        except Exception as e:  # noqa: BLE001
            print(f"  (grounding certificate skipped: {type(e).__name__}: {str(e)[:80]})")

        cert = (
            "# HermiT consistency certificate — `sdg-ontology`\n\n"
            f"- **isConsistent**: `{consistent}`\n"
            f"- **named classes**: {artifact_classes}\n"
            f"- **unsatisfiable classes**: {len(unsat)}\n"
            + (f"- **domain-narrowed**: {len(signals)} class(es) shed as unsatisfiable vs the theory — "
               f"justifications recorded in `build/realize_signals.json` (the re-authoring signal)\n" if signals else "")
            + f"- **realized from**: the {len(derived)} FinePDFs-derived templates (`catalog.json`)\n"
            f"- **rigor (Phase A)**: {nf} filler classes BFO-grounded · {na} classes carry NL definitions "
            f"(iao:0000115) · {nd} typed DataProperty assertions\n"
            + (f"- **individuals**: {n_inds} membrane-admitted (ABox included) — instance-level consistency "
               f"certified by DECOMPOSITION: nominal-free TBox + Types-only ABox ⇒ KB consistent ⟺ TBox "
               f"consistent ∧ every asserted type-conjunction satisfiable (each conjunction checked by "
               f"HermiT{f'; {n_withheld} withheld as clashing' if n_withheld else ''})\n" if n_inds else "")
            + "- **reasoner**: HermiT (OWLAPI, via DeepOnto)\n"
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
