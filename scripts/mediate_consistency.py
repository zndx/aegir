"""HermiT coherence check — the meta-harness's first GROUND-TRUTH gate (inc-2a).

DeepOnto's `Ontology(..., reasoner_type="hermit")` wraps HermiT — a sound &
complete OWL 2 DL reasoner (hypertableau). Our constructs are TBox (classes +
restrictions, no individuals), and a TBox with an *unsatisfiable* class is still
globally *consistent*, so the meaningful deductive check is COHERENCE: does the
construct introduce an unsatisfiable named class? We compose the construct with a
minimal BFO/CCO top-level disjointness grounding (Occurrent ⊥ Continuant) so a
construct anchored across disjoint categories yields an unsatisfiable head —
caught by `getUnsatisfiableClasses()`. Sound+complete ⇒ this verdict is ground
truth, not a proxy. (Cumulative cross-construct checking over the admitted set is
the next refinement; inc-2a checks the candidate against the grounding.)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import CatalogTemplate  # noqa: E402
from aegir.ontology.deeponto_harness import ensure_jvm, render_template_ontology  # noqa: E402

# Appended to the rendered construct (same bfo:/cco: prefixes render declares).
# Minimal BFO/CCO upper grounding: the seven anchors slotted under two disjoint
# top categories, so a cross-category construct is provably incoherent.
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


def coherence(template: CatalogTemplate) -> dict:
    """Return {consistent, is_consistent, unsat, error}. `consistent` = globally
    consistent AND no named class unsatisfiable (coherent). Fail-closed on
    render/reasoner error (an unreasonable construct is not admitted)."""
    ensure_jvm()
    from deeponto.onto import Ontology
    try:
        omn = render_template_ontology(template) + "\n" + _GROUNDING
    except Exception as e:
        return {"consistent": False, "is_consistent": False, "unsat": [], "error": f"render: {e}"}
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".omn", delete=False) as f:
            f.write(omn)
            path = f.name
        onto = Ontology(path, reasoner_type="hermit")
        r = onto.reasoner.owl_reasoner            # raw OWLAPI HermiT reasoner
        is_consistent = bool(r.isConsistent())
        unsat: list[str] = []
        if is_consistent:
            bottom = r.getUnsatisfiableClasses()   # Node<OWLClass>; bottom = owl:Nothing ∪ unsat
            for c in bottom.getEntities().toArray():
                if not c.isOWLNothing():
                    unsat.append(str(c.getIRI()))
        return {"consistent": bool(is_consistent and not unsat),
                "is_consistent": is_consistent, "unsat": unsat, "error": ""}
    except Exception as e:
        return {"consistent": False, "is_consistent": False, "unsat": [], "error": f"reason: {type(e).__name__}: {str(e)[:120]}"}
    finally:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_jvm()
    ok = CatalogTemplate(
        template_id="t_ok",
        manchester_template="Class: {X:Class} SubClassOf: cco:DescriptiveICE, {p:ObjectProperty} some {Y:Class}",
        slot_types={"X": "Class", "p": "ObjectProperty", "Y": "Class"},
        bfo_anchor_path=["cco:DescriptiveICE"])
    bad = CatalogTemplate(  # anchored across disjoint BFO categories → head unsatisfiable
        template_id="t_bad",
        manchester_template="Class: {X:Class} SubClassOf: bfo:Process, cco:Artifact",
        slot_types={"X": "Class"},
        bfo_anchor_path=["bfo:Process"])
    ro, rb = coherence(ok), coherence(bad)
    print(f"COHERENT   construct: {ro}")
    print(f"INCOHERENT construct: {rb}")
    assert ro["consistent"] is True, "well-formed construct must be coherent"
    assert rb["consistent"] is False and rb["unsat"], "cross-category construct must be incoherent (unsat head)"
    print("\nHermiT coherence gate verified: ground-truth reject of the incoherent construct, pass of the coherent one.")
