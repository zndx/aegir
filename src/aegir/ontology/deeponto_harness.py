"""DeepOnto offline harness — populates catalog metadata fields.

This module is **offline-only**. It is invoked at catalog construction
time (P1a per the v0.5 concept brief) to populate the
`is_complex`, `verbal_template`, and `mean_verbal_length` fields of
:class:`aegir.ontology.schema.CatalogTemplate` rows. The runtime
verifier and the v3 pretraining/inference paths are JVM-free —
they read the populated catalog by lookup and never call any code
in this module.

Usage:

    # set LD_LIBRARY_PATH so the system JDK 11 finds libz; see
    # `scripts/setup_jvm_env.sh` for the canonical bootstrap.
    from aegir.ontology.deeponto_harness import (
        ensure_jvm,
        probe_template,
    )
    ensure_jvm()
    result = probe_template(template)
    template.is_complex = result.is_complex
    template.verbal_template = result.verbal_template
    template.mean_verbal_length = result.mean_verbal_length

Two non-trivial gotchas inherited from gaius's prototype:

1.  DeepOnto's ``deeponto.onto`` module imports ``click.prompt()``
    at import time, which hangs in non-interactive environments
    if the JVM hasn't been started. Always call :func:`ensure_jvm`
    before importing any deeponto submodule.
2.  The system openjdk-11 ships a ``libzip.so`` that depends on
    ``libz.so.1``, but nix's devenv profile masks
    ``/usr/lib/x86_64-linux-gnu`` from the dynamic linker. The
    bootstrap script ``scripts/setup_jvm_env.sh`` creates a
    minimal ``/tmp/jvm-libs/`` directory containing only
    ``libz.so.1`` and points ``LD_LIBRARY_PATH`` at it — surgical
    enough to unblock the JVM without dragging in a system libc
    that conflicts with the venv's nix-built numpy/torch wheels.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from aegir.ontology.schema import CatalogTemplate

# Slot syntax matches `{name:Type}` and `{name:Type:Bound}`. Per
# `aegir.ontology.SLOT_DSL.md`. Importing the regex from
# `scripts/check_ontology_schema.py` would be cleaner once that
# logic is promoted into the package proper; for now this is
# duplicated.
SLOT_RE = re.compile(r"\{(?P<name>\w+):(?P<type>[\w:]+?)(?::(?P<bound>[\w:]+))?\}")

# Marker format for placeholder slot fillers. We use `ZZ<name>ZZ`
# because (a) spaCy's tokenizer in DeepOnto's verbalizer preserves
# this form verbatim across casing variations and (b) it is
# unlikely to appear in normal English. After verbalization,
# regex `ZZ(\w+?)ZZ` is reverse-mapped to `{<name>}` to produce
# the catalog's ``verbal_template`` field.
MARKER_PREFIX = "ZZ"
MARKER_SUFFIX = "ZZ"
MARKER_RE = re.compile(MARKER_PREFIX + r"(\w+?)" + MARKER_SUFFIX)

# Test-namespace URI used for placeholder filler IRIs during the
# offline pass. Confined to the harness; never appears in
# committed catalog content.
TEST_NAMESPACE = "http://aegir-test.example.org/sdg-test"

# Flag preventing repeated init_jvm calls within one process.
_JVM_STARTED = False

logger = logging.getLogger(__name__)


def ensure_jvm(memory: str | None = None) -> None:
    """Idempotent JVM init.

    Sets ``JVM_MEMORY`` env var (if not already set) before any
    deeponto import to avoid the click.prompt() hang documented
    in the module docstring.
    """
    global _JVM_STARTED
    if _JVM_STARTED:
        return
    mem = memory or os.environ.get("JVM_MEMORY", "4g")
    os.environ.setdefault("JVM_MEMORY", mem)

    import jpype  # noqa: F401  (forces a clear error if absent)

    if not jpype.isJVMStarted():
        from deeponto import init_jvm
        init_jvm(mem)
    _JVM_STARTED = True
    logger.info("JVM started with %s memory", mem)


# ---- Probe result ----


@dataclass
class ProbeResult:
    """Result of probing a single :class:`CatalogTemplate` against
    DeepOnto.

    Attributes:
        is_complex: ``True`` iff the rendered ontology has at least
            one entry in ``onto.get_asserted_complex_classes()``.
        verbal_template: Verbalization with slot variables
            (e.g. ``"something that {p} some {Y}"``) or empty
            string on failure.
        mean_verbal_length: Character length of ``verbal_template``
            after marker→slot substitution.
        loaded: ``True`` iff DeepOnto loaded the rendered ontology
            without raising.
        verbalized: ``True`` iff at least one verbalization
            succeeded with non-trivial output.
        error: Free-text error description on failure; empty on
            success.
        error_kind: One of ``"none"``, ``"render"``, ``"load"``,
            ``"verbalize"``, ``"unknown"``.
    """

    is_complex: bool = False
    verbal_template: str = ""
    mean_verbal_length: float = 0.0
    loaded: bool = False
    verbalized: bool = False
    error: str = ""
    error_kind: str = "none"
    diagnostics: dict[str, object] = field(default_factory=dict)


# ---- Render ----


def _slot_marker(slot_name: str) -> str:
    return f"{MARKER_PREFIX}{slot_name}{MARKER_SUFFIX}"


def _filler_iri(slot_name: str) -> str:
    return f"{TEST_NAMESPACE}#T_{slot_name}"


def render_template_ontology(template: CatalogTemplate) -> str:
    """Render a Manchester-syntax ontology document for a single
    template, with placeholder fillers substituted in.

    The slot→marker mapping is uniform (``ZZ<slot_name>ZZ``) and
    can be reconstructed by callers from the slot names.
    """
    slot_to_marker: dict[str, str] = {}
    slot_to_filler: dict[str, str] = {}

    declared_slots = list(template.slot_types.keys())
    for slot_name in declared_slots:
        slot_to_marker[slot_name] = _slot_marker(slot_name)
        slot_to_filler[slot_name] = _filler_iri(slot_name)

    rendered_axiom = template.manchester_template
    for m in SLOT_RE.finditer(template.manchester_template):
        slot_name = m.group("name")
        if slot_name not in slot_to_filler:
            continue
        rendered_axiom = rendered_axiom.replace(m.group(0), f"<{slot_to_filler[slot_name]}>")

    object_property_decls = []
    data_property_decls = []
    individual_decls = []
    class_decls: list[str] = []
    for slot_name, slot_type in template.slot_types.items():
        bare_type = slot_type.split(":")[0] if ":" in slot_type else slot_type
        marker = slot_to_marker[slot_name]
        filler = slot_to_filler[slot_name]
        decl = f'<{filler}>\n    Annotations: rdfs:label "{marker}"\n'
        if bare_type == "ObjectProperty":
            object_property_decls.append(f"ObjectProperty: {decl}")
        elif bare_type == "DataProperty":
            data_property_decls.append(f"DataProperty: {decl}")
        elif bare_type == "Individual":
            individual_decls.append(f"Individual: {decl}")
        else:
            class_decls.append(f"Class: {decl}")

    body = "\n".join(
        object_property_decls
        + data_property_decls
        + individual_decls
        + class_decls
        + [rendered_axiom]
    )

    extra_decls = _auto_declare_referenced_iris(template, set(slot_to_filler.values()))

    omn = (
        "Prefix: : <http://example.org/aegir-probe#>\n"
        "Prefix: owl: <http://www.w3.org/2002/07/owl#>\n"
        "Prefix: rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        "Prefix: skos: <http://www.w3.org/2004/02/skos/core#>\n"
        "Prefix: prov: <http://www.w3.org/ns/prov#>\n"
        "Prefix: dcterms: <http://purl.org/dc/terms/>\n"
        "Prefix: bfo: <http://purl.obolibrary.org/obo/BFO_>\n"
        "Prefix: cco: <http://www.commoncoreontologies.org/>\n"
        "Prefix: iao: <http://purl.obolibrary.org/obo/IAO_>\n"
        "Prefix: obi: <http://purl.obolibrary.org/obo/OBI_>\n"
        "Prefix: schema: <https://schema.org/>\n"
        "Prefix: dbo: <http://dbpedia.org/ontology/>\n"
        "Prefix: sdg: <https://signals360.example.org/sdg#>\n"
        "\n"
        "Ontology: <http://example.org/aegir-probe>\n"
        "\n"
        + body
        + "\n"
        + extra_decls
    )
    return omn


# Manchester-syntax keywords we strip from token scans.
MANCHESTER_KEYWORDS = {
    "Class:", "SubClassOf:", "EquivalentTo:", "DisjointWith:",
    "ObjectProperty:", "DataProperty:", "Individual:",
    "Annotations:", "Domain:", "Range:", "Characteristics:",
    "Functional", "InverseFunctional", "Symmetric", "Asymmetric",
    "Reflexive", "Irreflexive", "Transitive",
    "and", "or", "not", "some", "only", "min", "max", "exactly",
    "value", "Self", "inverse",
}

# Recognized public-namespace prefixes; anything starting with these
# is treated as an IRI to potentially auto-declare.
KNOWN_PREFIXES = ("bfo:", "cco:", "iao:", "obi:", "schema:", "dbo:",
                  "sdg:", "skos:", "prov:", "dcterms:", "owl:",
                  "rdf:", "rdfs:", "xsd:")

# CURIE matcher for ``prefix:LocalName`` (or ``prefix:0000123`` for
# BFO-style numeric IRIs).
CURIE_RE = re.compile(r"\b(?:" + "|".join(KNOWN_PREFIXES) + r")[\w\-]+\b")

# Common public-namespace IRIs whose canonical labels we ship so
# auto-declared classes/properties verbalize as natural English
# rather than as opaque local-name fragments. Sourced from BFO
# 2020 and CCO public release labels.
PUBLIC_LABELS: dict[str, str] = {
    # BFO 2020 (numeric IRIs)
    "bfo:0000001": "entity",
    "bfo:0000002": "continuant",
    "bfo:0000003": "occurrent",
    "bfo:0000004": "independent continuant",
    "bfo:0000015": "process",
    "bfo:0000017": "realizable entity",
    "bfo:0000019": "quality",
    "bfo:0000020": "specifically dependent continuant",
    "bfo:0000023": "role",
    "bfo:0000024": "fiat object part",
    "bfo:0000027": "object aggregate",
    "bfo:0000029": "site",
    "bfo:0000030": "object",
    "bfo:0000031": "generically dependent continuant",
    "bfo:0000034": "function",
    "bfo:0000040": "material entity",
    "bfo:0000050": "part of",
    "bfo:0000051": "has part",
    "bfo:0000054": "realized in",
    "bfo:0000055": "realizes",
    "bfo:0000056": "participates in",
    "bfo:0000057": "has participant",
    "bfo:0000063": "precedes",
    "bfo:0000067": "contains process",
    "bfo:0000108": "exists at",
    "bfo:0000182": "history",
    # CCO common
    "cco:Artifact": "artifact",
    "cco:Person": "person",
    "cco:Organization": "organization",
    "cco:InformationContentEntity": "information content entity",
    "cco:DesignativeICE": "designative information content entity",
    "cco:DescriptiveICE": "descriptive information content entity",
    "cco:DirectiveICE": "directive information content entity",
}


def _auto_declare_referenced_iris(template: CatalogTemplate, slot_filler_iris: set[str]) -> str:
    """Emit Class/ObjectProperty stubs for any IRI in the template
    body that isn't a slot.

    This is a best-effort heuristic for DeepOnto's verbalizer, which
    silently skips entities it doesn't see in the loaded ontology.
    Auto-declaration ensures the verbalizer can resolve labels for
    IRIs hardcoded in the template (e.g., ``cco:Artifact``).

    The heuristic for ObjectProperty vs Class: tokens that appear
    immediately before ``some|only|min|max|exactly|value`` are
    ObjectProperty; everything else is Class. This is approximate
    but handles the common axiom shapes.
    """
    # Find the substituted template body would be — but we operate
    # on the raw manchester_template, since slots in it may also
    # be CURIE-shaped (rare but possible).
    body = template.manchester_template

    # Find all CURIE references; exclude slot-pattern matches.
    references: list[str] = []
    for m in CURIE_RE.finditer(body):
        token = m.group(0)
        if token in MANCHESTER_KEYWORDS:
            continue
        references.append(token)

    if not references:
        return ""

    # Heuristic: if a token appears just before a restriction keyword,
    # it's an ObjectProperty.
    restriction_kw = {"some", "only", "min", "max", "exactly", "value"}
    object_property_tokens: set[str] = set()
    class_tokens: set[str] = set()
    tokens = body.split()
    for i, tok in enumerate(tokens):
        clean = tok.rstrip(",.()")
        if clean not in references:
            continue
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
        if next_tok in restriction_kw:
            object_property_tokens.add(clean)
        else:
            class_tokens.add(clean)

    def _local_label(curie: str) -> str:
        """Derive a human-readable label from a CURIE.

        Looks up canonical labels in :data:`PUBLIC_LABELS` first
        (handles BFO numeric IRIs and CCO upper structure where
        local-name extraction would produce useless fragments).
        Falls back to a CamelCase / underscore / hyphen split of
        the local name for IRIs not in the dictionary.
        """
        if curie in PUBLIC_LABELS:
            return PUBLIC_LABELS[curie]
        local = curie.split(":", 1)[1] if ":" in curie else curie
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
        spaced = spaced.replace("_", " ").replace("-", " ")
        return spaced.lower()

    decls: list[str] = []
    for tok in sorted(object_property_tokens):
        label = _local_label(tok)
        decls.append(
            f"ObjectProperty: {tok}\n"
            f"    Annotations: rdfs:label \"{label}\"\n"
        )
    for tok in sorted(class_tokens):
        if tok in {"owl:Thing", "owl:Nothing"}:
            continue
        label = _local_label(tok)
        decls.append(
            f"Class: {tok}\n"
            f"    Annotations: rdfs:label \"{label}\"\n"
        )
    return "\n".join(decls)


# ---- Verbalize ----


def _markers_to_slots(verbal: str) -> str:
    """Replace ``ZZ<name>ZZ`` markers in a verbalization with
    ``{<name>}`` slot placeholders."""
    return MARKER_RE.sub(lambda m: "{" + m.group(1) + "}", verbal)


def _marker_count(verbal: str) -> int:
    """Count the distinct ``ZZ<name>ZZ`` slot markers in a
    pre-substitution verbalization. Used by the candidate
    selector in :func:`_verbalize_for_template` to prefer
    slot-bearing restriction verbalizations over parent-class
    verbalizations whose body has no slots.
    """
    return len(set(MARKER_RE.findall(verbal)))


def _head_slot_name(template: CatalogTemplate) -> str | None:
    """Identify the template's *head* slot — the first
    ``{<name>:Class}`` placeholder.

    Returns the slot name or ``None`` if the template has no
    Class-typed slot at all (degenerate).
    """
    for m in SLOT_RE.finditer(template.manchester_template):
        if m.group("type") == "Class":
            return m.group("name")
    return None


def _verbalize_for_template(
    onto,
    verbaliser,
    head_iri: str,
    head_slot_name: str,
) -> tuple[str, str]:
    """Run the appropriate verbalizer call(s) on the rendered
    ontology and return ``(verbal_template, error)``.

    Tries multiple axioms involving the head class and accumulates
    successful verbalizations. Returns the longest-content
    verbalization (heuristic: more detail = more informative).
    Falls through gracefully when individual verbalizer calls
    raise (e.g., DeepOnto's verbaliser lacks support for
    ObjectMinCardinality / ObjectMaxCardinality /
    ObjectExactCardinality expressions, so we silently skip
    those expressions and try the next axiom strategy).
    """
    candidates: list[str] = []
    last_error = ""

    # 1. Equivalence axioms.
    for ax in onto.get_equivalence_axioms("Classes"):
        ax_str = str(ax)
        if head_iri not in ax_str:
            continue
        try:
            out = verbaliser.verbalise_class_equivalence_axiom(ax)
            if isinstance(out, tuple) and len(out) == 2:
                sub_v = out[0].verbal if hasattr(out[0], "verbal") else str(out[0])
                sup_v = out[1].verbal if hasattr(out[1], "verbal") else str(out[1])
                candidates.append(f"{sub_v} is equivalent to {sup_v}")
            elif hasattr(out, "verbal"):
                candidates.append(out.verbal)
            else:
                candidates.append(str(out))
        except Exception as e:
            last_error = f"equivalence verbalize: {e}"

    # 2. Subsumption axioms whose superclass is a complex expression.
    complex_classes = list(onto.get_asserted_complex_classes())
    sub_axioms = list(onto.get_subsumption_axioms("Classes"))

    for ax in sub_axioms:
        ax_str = str(ax)
        if head_iri not in ax_str:
            continue
        # Try complex-expression verbalization on each complex class
        # appearing in this axiom. Failures here (notably for
        # cardinality restrictions, which DeepOnto's verbaliser
        # does not support) are recorded but not papered over with
        # hand-rolled verbalizations — manufacturing a fallback
        # would inject spurious signal into the downstream RL
        # reward, degrading the verifier's discriminative validity.
        head_marker = MARKER_PREFIX + head_slot_name + MARKER_SUFFIX
        complex_succeeded_for_this_axiom = False
        for cc in complex_classes:
            cc_str = str(cc)
            if cc_str not in ax_str:
                continue
            try:
                out = verbaliser.verbalise_class_expression(cc)
                body = out.verbal if hasattr(out, "verbal") else str(out)
                # Keep ``head_marker`` in the candidate (rather than
                # substituting ``{<name>}`` early) so the slot-count
                # ranker in the selection step counts the head slot
                # alongside whatever Y/Z markers ``body`` carries.
                candidates.append(f"{head_marker} is {body}")
                complex_succeeded_for_this_axiom = True
            except Exception as e:
                last_error = f"complex expression verbalize: {e}"

        # Add the subsumption-axiom verbalization only when the
        # complex-expression form did NOT succeed for this axiom.
        # When it did, the subsumption form would be a redundant
        # ``X is a something that ...`` paraphrase of the complex
        # form. Falling back here is what enables cardinality
        # axioms (which the complex verbaliser rejects) to still
        # produce a candidate via the basic ``X is a Y`` shape.
        if not complex_succeeded_for_this_axiom:
            try:
                out = verbaliser.verbalise_class_subsumption_axiom(ax)
                if isinstance(out, tuple) and len(out) == 2:
                    sub_v = out[0].verbal if hasattr(out[0], "verbal") else str(out[0])
                    sup_v = out[1].verbal if hasattr(out[1], "verbal") else str(out[1])
                    candidates.append(f"{sub_v} is a {sup_v}")
                elif hasattr(out, "verbal"):
                    candidates.append(out.verbal)
                else:
                    candidates.append(str(out))
            except Exception as e:
                last_error = f"subsumption verbalize: {e}"

    if not candidates:
        return "", last_error or "no verbalizable axiom found for head class"

    # Prefer verbalizations with more slot-marker placeholders
    # (richer per-template content for downstream R_C / R_D);
    # ties broken by *longer* length so the more semantically
    # informative form wins among same-slot-count candidates
    # (e.g. ``X is something that verifies <directive>`` beats
    # ``X is a process`` for templates whose target axiom is the
    # restriction). Pre-substitution markers ZZ<name>ZZ are still
    # in place at this point — count those, not post-substitution
    # braces. The complex-vs-subsumption duplicate is avoided at
    # candidate-generation time, so we won't see the awkward
    # ``X is a something that ...`` form against a same-shape
    # complex candidate.
    candidates.sort(key=lambda v: (-_marker_count(v), -len(v)))
    chosen = candidates[0]
    return _markers_to_slots(chosen), ""


# ---- Probe ----


def probe_template(template: CatalogTemplate) -> ProbeResult:
    """Run the offline DeepOnto pass on a single catalog template.

    Renders the template into a tiny Manchester-syntax ontology,
    loads it via DeepOnto, populates ``is_complex`` from
    ``get_asserted_complex_classes()``, runs the verbalizer, and
    derives ``verbal_template`` and ``mean_verbal_length``.

    Caller is responsible for calling :func:`ensure_jvm` first.
    """
    if not _JVM_STARTED:
        return ProbeResult(error="JVM not started; call ensure_jvm() first", error_kind="unknown")

    head_slot_name = _head_slot_name(template)
    if head_slot_name is None:
        return ProbeResult(
            error="template has no Class-typed slot; cannot identify head",
            error_kind="render",
        )

    try:
        omn_content = render_template_ontology(template)
    except Exception as e:
        return ProbeResult(error=f"render failure: {e}", error_kind="render")

    try:
        from deeponto.onto import Ontology, OntologyVerbaliser
    except Exception as e:
        return ProbeResult(error=f"deeponto import failure: {e}", error_kind="unknown")

    omn_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".omn",
            delete=False,
        ) as f:
            f.write(omn_content)
            omn_path = Path(f.name)

        try:
            onto = Ontology(str(omn_path))
        except Exception as e:
            return ProbeResult(
                error=f"DeepOnto load failure: {e}",
                error_kind="load",
                diagnostics={"omn_excerpt": omn_content[:400]},
            )

        is_complex = bool(list(onto.get_asserted_complex_classes()))

        verbaliser = OntologyVerbaliser(onto)
        head_iri = _filler_iri(head_slot_name)
        verbal_template, verb_err = _verbalize_for_template(
            onto, verbaliser, head_iri, head_slot_name
        )

        if verbal_template:
            mean_verbal_length = float(len(verbal_template))
            return ProbeResult(
                is_complex=is_complex,
                verbal_template=verbal_template,
                mean_verbal_length=mean_verbal_length,
                loaded=True,
                verbalized=True,
                error="",
                error_kind="none",
                diagnostics={
                    "head_slot": head_slot_name,
                    "head_iri": head_iri,
                    "n_complex_asserted": len(list(onto.get_asserted_complex_classes())),
                },
            )
        return ProbeResult(
            is_complex=is_complex,
            verbal_template="",
            mean_verbal_length=0.0,
            loaded=True,
            verbalized=False,
            error=verb_err or "no verbalizable axiom",
            error_kind="verbalize",
            diagnostics={"head_slot": head_slot_name, "head_iri": head_iri},
        )

    finally:
        if omn_path is not None and omn_path.exists():
            omn_path.unlink(missing_ok=True)
