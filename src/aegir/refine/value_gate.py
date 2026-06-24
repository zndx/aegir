"""Refinement-loop VALUE_GATE (prototype) — value-level HermiT consistency.

The membrane's sharpest gate. Given a column typed at an ontology concept ``C`` and a set of cell values (each
carrying its *source* concept from the entity-value pools), check whether the values are ADMISSIBLE under
``C`` — i.e. no value's source concept is DISJOINT from ``C``, directly OR by inference. This is what rejects
the audit's "ASHRAE 62.1 in a medical-imaging column" instead of letting the LLM rationalize it: ASHRAE's
source (an HVAC standard) is disjoint from imaging-protocol, so asserting one individual in both classes
yields an ontology HermiT proves inconsistent.

Why HermiT and not a flat disjointness lookup: the disjointness is usually stated at a PARENT level
(`clinical_descriptor ⊥ facility_descriptor`) and must be *inferred* down to the leaves
(`imaging_protocol ⊑ clinical_descriptor`, `hvac_standard ⊑ facility_descriptor`). The reasoner computes that
closure; a lookup on the leaf pair would miss it.

Prototype scope: the value-ontology fragment (classes + subsumption + disjointness) is supplied per call.
inc-1 bootstraps the full fragment from the (refined) domain taxonomy + the deriver→HermiT admission
machinery. Runs HermiT via deeponto, reusing ``reasoning_gates.ensure_jvm`` — needs
``LD_LIBRARY_PATH=build/jvm-libs``.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

_HEADER = ("Prefix: : <http://example.org/aegir-value#>\n"
           "Ontology: <http://example.org/aegir-value>\n\n")


def _iri(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name)).strip("_")
    return s or "x"


def render_value_ontology(classes, subclass_of, disjoint, individuals) -> str:
    """Manchester doc: class decls + ``SubClassOf`` + ``DisjointClasses`` + ``Individual … Types: …``."""
    lines = [_HEADER]
    for c in sorted(set(classes)):
        lines.append(f"Class: {_iri(c)}")
    for sub, sup in subclass_of:
        lines.append(f"Class: {_iri(sub)}\n    SubClassOf: {_iri(sup)}")
    for grp in disjoint:   # each group is mutually disjoint (n-ary); a 2-list is the pair case
        lines.append("DisjointClasses: " + ", ".join(_iri(x) for x in grp))
    for iid, types in individuals:
        lines.append(f"Individual: {_iri(iid)}\n    Types: " + ", ".join(_iri(t) for t in types))
    return "\n".join(lines) + "\n"


def _is_consistent(doc: str) -> bool:
    from aegir.ontology.reasoning_gates import ensure_jvm
    ensure_jvm()
    from deeponto.onto import Ontology
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
            f.write(doc)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        return bool(onto.reasoner.owl_reasoner.isConsistent())
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def check_column(column_concept: str, value_sources, *, classes=(), subclass_of=(), disjoint=()) -> dict:
    """Gate a column's values against its concept.

    ``value_sources``: ``[(value_id, source_concept)]`` — the source concept is where the value came from in
    the entity-value pools (its true type). Each value is asserted ``Types: column_concept, source_concept``;
    a value whose source is disjoint from ``column_concept`` (directly or inferred) makes the batch
    inconsistent. Returns ``{consistent, violations: [value_id]}`` — on a failed batch, the offending values
    are pinpointed per-value (cheap at chapter scope; inc-1 batches + uses explanations at corpus scope)."""
    allcls = set(classes) | {column_concept} | {s for _, s in value_sources}
    inds = [(v, [column_concept, s]) for v, s in value_sources]
    if _is_consistent(render_value_ontology(allcls, subclass_of, disjoint, inds)):
        return {"consistent": True, "violations": []}
    viol = []
    for v, s in value_sources:
        doc = render_value_ontology(allcls, subclass_of, disjoint, [(v, [column_concept, s])])
        if not _is_consistent(doc):
            viol.append(v)
    return {"consistent": False, "violations": viol}


def check_chapter(columns, *, classes=(), subclass_of=(), disjoint=()) -> dict:
    """Batched whole-chapter value check — ONE HermiT classification over every column's cells (each a unique
    individual ``Types: column_concept, source``). On a clean chapter this is a single reasoner call; only if
    the batch is inconsistent do we fall back to per-column ``check_column`` to pinpoint the violators. Lets the
    refinement loop gate a full live chapter (dozens of columns) without a per-column reasoner storm.

    ``columns``: ``[(col_id, column_concept, [(value, source)])]``. Returns ``{consistent, violations:[col_id:value]}``."""
    allcls = set(classes)
    inds = []
    for col_id, concept, vs in columns:
        allcls |= {concept} | {s for _, s in vs}
        for i, (v, s) in enumerate(vs):
            inds.append((f"{col_id}__{i}", [concept, s]))
    if not inds or _is_consistent(render_value_ontology(allcls, subclass_of, disjoint, inds)):
        return {"consistent": True, "violations": []}
    viol = []
    for col_id, concept, vs in columns:
        r = check_column(concept, vs, classes=classes, subclass_of=subclass_of, disjoint=disjoint)
        viol.extend(f"{col_id}:{v}" for v in r["violations"])
    return {"consistent": False, "violations": viol}


if __name__ == "__main__":
    # The audit case — disjointness stated at the PARENT level, so the leaf-level violation is INFERRED.
    onto = dict(
        classes=["imaging_protocol", "hvac_standard", "clinical_descriptor", "facility_descriptor"],
        subclass_of=[("imaging_protocol", "clinical_descriptor"),
                     ("hvac_standard", "facility_descriptor")],
        disjoint=[("clinical_descriptor", "facility_descriptor")],   # parent-level disjointness
    )
    contaminated = [("t2_weighted_fat_sat", "imaging_protocol"),
                    ("gradient_echo_t2", "imaging_protocol"),
                    ("ashrae_62_1", "hvac_standard"),        # the contaminant the audit found
                    ("epa_to_17", "hvac_standard")]
    clean = [("t2_weighted_fat_sat", "imaging_protocol"),
             ("gradient_echo_t2", "imaging_protocol"),
             ("coronal_t2_fse", "imaging_protocol")]
    print("CONTAMINATED imaging column:", check_column("imaging_protocol", contaminated, **onto))
    print("CLEAN imaging column:       ", check_column("imaging_protocol", clean, **onto))
