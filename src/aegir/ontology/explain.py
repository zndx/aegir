"""explain.py — the reasoner-boundary SIGNAL machinery (Holland, Signals & Boundaries).

The realize's HermiT check is the BOUNDARY where a full-context logical conflict first becomes visible. On its
own it emits only a NAME — "X is unsatisfiable" — a signal no agent can adapt to. This module makes the boundary
emit the JUSTIFICATION instead: the MINIMAL set of axioms that force the class empty, rendered legibly, so the
LLM SEES why and can re-author the offending conjunct. That is the machinery that lets the signal cross the
boundary into the agent.

Uses the clarkparsia BlackBox/HST explanation generator over HermiT (both on the DeepOnto classpath). The caller
must have started the JVM (``ensure_jvm``) and hold the ``OWLOntology`` + its ``OWLReasoner`` from the same
manager — i.e. call it from inside the realize's ``_reason`` where the unsatisfiable, full-context ontology
already exists.
"""
from __future__ import annotations

import re


def _readable(axiom: str) -> str:
    """Collapse full IRIs to local names so the axiom reads to the agent (and to a human)."""
    axiom = re.sub(r"<[^>]*[#/]([A-Za-z0-9_]+)>", r"\1", axiom)
    return re.sub(r"\s+", " ", axiom).strip()


def _why(axioms: "list[str]") -> str:
    """A one-line, agent-legible cause read off the justification's shape (best-effort over the common cases —
    the role/realization and dual-grounding collisions the deriver actually produces)."""
    disj = [a for a in axioms if a.startswith("DisjointClasses")]
    rng = [a for a in axioms if a.startswith("ObjectPropertyRange")]
    if disj and rng:
        return ("a property range forces a filler into a category disjoint from its grounding — re-author the "
                "conjunct so the filler's type matches the range (e.g. realize a role in a Process/occurrent, "
                "not a Function/continuant)")
    if disj:
        return "the class is subsumed by two disjoint BFO/CCO categories — drop or re-ground one conjunct"
    return "unsatisfiable against the theory (BFO + π(CCO)); re-author the conflicting conjunct shown above"


def explain_unsatisfiable(owl_onto, reasoner, class_iris, cap: int = 40) -> "dict[str, dict]":
    """For each unsatisfiable class IRI, return ``{iri: {"axioms": [readable justification], "why": <summary>}}``
    — the minimal axioms entailing its emptiness (the SIGNAL the agent re-authors against). ``owl_onto`` is the
    OWLOntology, ``reasoner`` its OWLReasoner (both live in the caller's manager); ``cap`` bounds how many
    classes are explained (justification search is O(axioms) per class)."""
    import jpype
    HermiTFactory = jpype.JClass("org.semanticweb.HermiT.ReasonerFactory")
    BlackBox = jpype.JClass("com.clarkparsia.owlapi.explanation.BlackBoxExplanation")
    HST = jpype.JClass("com.clarkparsia.owlapi.explanation.HSTExplanationGenerator")
    IRI = jpype.JClass("org.semanticweb.owlapi.model.IRI")

    df = owl_onto.getOWLOntologyManager().getOWLDataFactory()
    gen = HST(BlackBox(owl_onto, HermiTFactory(), reasoner))
    out: dict = {}
    for iri in list(class_iris)[:cap]:
        try:
            cls = df.getOWLClass(IRI.create(jpype.JString(str(iri))))
            just = gen.getExplanation(cls)
            axioms = sorted(_readable(str(a)) for a in just.toArray())
            out[str(iri)] = {"axioms": axioms, "why": _why(axioms)}
        except Exception as e:  # noqa: BLE001 — a failed explanation must not crash the boundary
            out[str(iri)] = {"axioms": [], "why": f"(justification unavailable: {type(e).__name__})"}
    return out
