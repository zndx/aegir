"""Reasoning gates over DeepOnto + HermiT — the deep membrane, AGGREGATE and single-pass.

Extends the CPU membrane (parse / complex-class / verbalization) and the per-template coherence check
(``scripts/mediate_consistency``) with reasoner-derived gates over the WHOLE candidate batch at once:

  * **Gate 4 — consistency + coherence (HARD).** Render (BFO/CCO disjointness grounding ∪ every candidate
    head) into one ontology, classify ONCE with HermiT, require ``isConsistent`` AND zero unsatisfiable
    named classes. Catches cross-primitive contradictions a per-template check cannot (two individually
    coherent primitives that are jointly unsatisfiable) and is *cheaper on a tinybox* — one classification,
    not N. Unsatisfiable IRIs map back to the offending primitives so they can be dropped.
  * **Gate 5a — equivalence dedup (HARD).** If HermiT proves a candidate head ≡ another named class, it is
    redundant → drop. An entailment-based dedup, stronger than the structural ``_canon_key`` merge.
  * **Gate 5b/5c — inference-depth + taxonomy (DIAGNOSTIC, not a hard threshold).** Count reasoner-inferred
    superclasses per head (a richness signal — non-trivial only when axioms interact: definitions /
    disjointness / property chains) and report taxonomy size. NOT gated: over a fixed BFO/CCO backbone a
    thin ``SubClassOf … some`` legitimately triggers no reclassification and the global depth is the
    backbone's regardless of primitive quality, so an arbitrary depth>N threshold is non-discriminative.

All four read off the SAME classified reasoner — load the union once, query many. Fail-closed on
render/reasoner error (an un-reasonable batch is not admitted). Uses the raw OWLAPI HermiT reasoner via
``onto.reasoner.owl_reasoner`` (the path proven in inc-2a).
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from aegir.ontology.deeponto_harness import (TEST_NAMESPACE, _head_slot_name, ensure_jvm,
                                             render_template_ontology)
from aegir.ontology.schema import CatalogTemplate
# Minimal BFO/CCO upper grounding (mirrors mediate_consistency): the anchors under two disjoint top
# categories, so a construct spanning Occurrent⊥Continuant yields an unsatisfiable head.
_GROUNDING = """
Class: bfo:Occurrent
Class: bfo:Continuant
DisjointClasses: bfo:Occurrent, bfo:Continuant
Class: bfo:Process SubClassOf: bfo:Occurrent
Class: bfo:IndependentContinuant SubClassOf: bfo:Continuant
Class: cco:Artifact SubClassOf: bfo:IndependentContinuant
Class: cco:InformationContentEntity SubClassOf: bfo:Continuant
Class: cco:DescriptiveICE SubClassOf: cco:InformationContentEntity
Class: cco:DirectiveICE SubClassOf: cco:InformationContentEntity
Class: cco:DesignativeICE SubClassOf: cco:InformationContentEntity
"""
_FILLER_RE = re.compile(re.escape(TEST_NAMESPACE) + r"#T_([A-Za-z0-9_]+)")


def _tid(template: CatalogTemplate) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", template.template_id)


def render_batch(templates: "list[CatalogTemplate]") -> "tuple[str, dict[str, str]]":
    """One OMN document: the BFO/CCO grounding + every renderable candidate, with each template's probe
    IRIs namespaced by template_id (so heads/fillers stay distinct; shared bfo:/cco:/sdg: vocab is NOT
    namespaced, so the grounding + cross-primitive interactions apply). Returns (omn, head_iri_by_tid)."""
    header, bodies, head_iri = "", [], {}
    for t in templates:
        try:
            omn = render_template_ontology(t)
        except Exception:  # noqa: BLE001 — unrenderable → skip (it fails G1 anyway)
            continue
        cut = omn.find("Ontology:")
        if not header:
            header = omn[:cut]
        body = omn[cut:].split("\n", 1)[1] if cut >= 0 else omn
        tid = _tid(t)
        body = _FILLER_RE.sub(lambda m, _t=tid: f"{TEST_NAMESPACE}#T_{_t}__{m.group(1)}", body)
        bodies.append(body)
        hs = _head_slot_name(t)
        if hs:
            head_iri[t.template_id] = f"{TEST_NAMESPACE}#T_{tid}__{hs}"
    if not header:
        header = "Prefix: : <http://example.org/aegir-probe#>\nPrefix: owl: <http://www.w3.org/2002/07/owl#>\n"
    doc = header + "Ontology: <http://example.org/aegir-batch>\n\n" + _GROUNDING + "\n" + "\n".join(bodies)
    return doc, head_iri


def reason(templates: "list[CatalogTemplate]") -> dict:
    """Classify the batch once and read off all reasoner gates. Returns:
    {consistent, unsat (tids), equivalent (tid->[iris]), inferred (tid->n_supers), n_classes, error}."""
    ensure_jvm()
    from deeponto.onto import Ontology
    doc, head_iri = render_batch(templates)
    if not head_iri:
        return {"consistent": False, "unsat": [], "equivalent": {}, "inferred": {}, "n_classes": 0,
                "error": "no renderable candidates"}
    iri_to_tid = {v: k for k, v in head_iri.items()}
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".omn", delete=False) as f:
            f.write(doc)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        r = onto.reasoner.owl_reasoner
        consistent = bool(r.isConsistent())
        out = {"consistent": consistent, "unsat": [], "equivalent": {}, "inferred": {},
               "n_classes": 0, "error": ""}
        if not consistent:
            out["error"] = "globally inconsistent batch"
            return out
        # Gate 4: unsatisfiable named heads
        bottom = r.getUnsatisfiableClasses()
        unsat_iris = {str(c.getIRI()) for c in bottom.getEntities().toArray() if not c.isOWLNothing()}
        out["unsat"] = sorted({iri_to_tid[i] for i in unsat_iris if i in iri_to_tid})
        # look up head OWLClass objects for the per-head queries
        classes = getattr(onto, "owl_classes", {}) or {}
        out["n_classes"] = len(classes)
        for tid, iri in head_iri.items():
            cls = classes.get(iri)
            if cls is None:
                continue
            try:  # Gate 5a: equivalence dedup
                eq = [str(e.getIRI()) for e in r.getEquivalentClasses(cls).getEntities().toArray()
                      if str(e.getIRI()) != iri and not e.isOWLNothing() and not e.isOWLThing()]
                if eq:
                    out["equivalent"][tid] = eq
            except Exception:  # noqa: BLE001
                pass
            try:  # Gate 5b: inferred superclasses (diagnostic)
                sup = r.getSuperClasses(cls, False)
                out["inferred"][tid] = len([s for s in sup.getFlattened().toArray() if not s.isOWLThing()])
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception as e:  # noqa: BLE001
        return {"consistent": False, "unsat": [], "equivalent": {}, "inferred": {}, "n_classes": 0,
                "error": f"reason: {type(e).__name__}: {str(e)[:160]}"}
    finally:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)


def batch_gate(templates: "list[CatalogTemplate]") -> dict:
    """Apply the reasoning gates → which templates SURVIVE (consistent ∧ satisfiable ∧ non-redundant) +
    diagnostics. Equivalence is resolved greedily: the first of an equivalent pair is kept."""
    res = reason(templates)
    survivors, dropped = [], {}
    if not res["consistent"]:
        return {"survivors": [], "dropped": {t.template_id: "inconsistent-batch" for t in templates},
                "reason": res, "consistent": False}
    unsat = set(res["unsat"])
    seen_equiv: set[str] = set()
    for t in templates:
        tid = t.template_id
        if tid in unsat:
            dropped[tid] = "unsatisfiable"
            continue
        eq = res["equivalent"].get(tid)
        if eq and any(e in seen_equiv for e in eq):
            dropped[tid] = f"equivalent-to {eq}"
            continue
        survivors.append(t)
        seen_equiv.add(head_iri_of(t))
    # inference-depth diagnostic over SURVIVORS only — an unsatisfiable class trivially subclasses
    # everything (its 19-super signature), which would otherwise inflate the richness signal.
    inferred = res["inferred"]
    surv_inf = [inferred[t.template_id] for t in survivors if t.template_id in inferred]
    mean_inf = round(sum(surv_inf) / max(1, len(surv_inf)), 2)
    return {"survivors": survivors, "dropped": dropped, "consistent": True,
            "n_classes": res["n_classes"], "mean_inferred_supers": mean_inf, "reason": res}


def head_iri_of(template: CatalogTemplate) -> str:
    hs = _head_slot_name(template)
    return f"{TEST_NAMESPACE}#T_{_tid(template)}__{hs}" if hs else ""
