"""ABox bridge + HermiT realization — realization-as-CPA beachhead (inc-2d).

The descoped G-rel claim (a tiny model can't learn relational/type skill — it floored)
is RE-HOMED to the reasoner here. A generated corpus table is lifted to an OWL **ABox**
(individuals + object-property facts from FK edges); HermiT **realization** then
*computes* each individual's most-specific types and role fillers. Because the model is
sound & complete, those computed types ARE the column/entity annotations (CPA) — not a
learned proxy. The byte model becomes a fast amortization of this, not the thing that
must learn it.

This module is the primitive layer + a standalone proof that realization computes an
UNASSERTED type (via object-property Domain/Range — the relational inference G-rel was
about). The corpus-scale bridge (rows → ABox → label_idx over the 540 templates →
eval_ontology_cpa) is inc-2d-full; see docs/scratch/2026-06-16/051833_inc2d_*.md.

Raw OWLAPI via jpype (DeepOnto does not wrap ABox assertion). After any ABox mutation
the reasoner MUST be reloaded (HermiT classifies a snapshot) — see `refresh`.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.deeponto_harness import ensure_jvm  # noqa: E402


def _owlapi():
    """Import the OWLAPI model classes (JVM must be started: call ensure_jvm first)."""
    from org.semanticweb.owlapi.model import IRI  # type: ignore
    return IRI


def _ind(onto, iri: str):
    return onto.owl_data_factory.getOWLNamedIndividual(_owlapi().create(iri))


def _cls(onto, iri: str):
    return onto.owl_data_factory.getOWLClass(_owlapi().create(iri))


def _obj_prop(onto, iri: str):
    return onto.owl_data_factory.getOWLObjectProperty(_owlapi().create(iri))


def _data_prop(onto, iri: str):
    return onto.owl_data_factory.getOWLDataProperty(_owlapi().create(iri))


def assert_type(onto, ind_iri: str, class_iri: str) -> None:
    f = onto.owl_data_factory
    onto.add_axiom(f.getOWLClassAssertionAxiom(_cls(onto, class_iri), _ind(onto, ind_iri)),
                   return_undo=False)


def assert_relation(onto, subj_iri: str, prop_iri: str, obj_iri: str) -> None:
    f = onto.owl_data_factory
    onto.add_axiom(f.getOWLObjectPropertyAssertionAxiom(
        _obj_prop(onto, prop_iri), _ind(onto, subj_iri), _ind(onto, obj_iri)), return_undo=False)


def assert_data(onto, ind_iri: str, prop_iri: str, value, datatype: str | None = None) -> None:
    """Assert a data-property fact. `value` is typed: an int/float → numeric literal,
    else a plain/typed string literal (datatype = an xsd IRI string, optional)."""
    f = onto.owl_data_factory
    if isinstance(value, bool):
        lit = f.getOWLLiteral(value)
    elif isinstance(value, int):
        lit = f.getOWLLiteral(int(value))
    elif isinstance(value, float):
        lit = f.getOWLLiteral(float(value))
    elif datatype:
        lit = f.getOWLLiteral(str(value), f.getOWLDatatype(_owlapi().create(datatype)))
    else:
        lit = f.getOWLLiteral(str(value))
    onto.add_axiom(f.getOWLDataPropertyAssertionAxiom(_data_prop(onto, prop_iri),
                   _ind(onto, ind_iri), lit), return_undo=False)


def refresh(onto) -> None:
    """Re-create the reasoner over the mutated ontology. HermiT classifies a snapshot,
    so newly-asserted ABox axioms are invisible until this is called."""
    onto.reasoner.load_reasoner(onto.reasoner_type)


def realize_types(onto, ind_iri: str, direct: bool = True) -> list[str]:
    """The CPA computation: HermiT's inferred class IRIs for an individual (most-specific
    when direct=True). owl:Thing dropped. Call `refresh` after asserting first."""
    node_set = onto.reasoner.owl_reasoner.getTypes(_ind(onto, ind_iri), direct)
    out = []
    for c in node_set.getFlattened():
        if not c.isOWLThing():
            out.append(str(c.getIRI()))
    return out


def realize_relations(onto, subj_iri: str, prop_iri: str) -> list[str]:
    """HermiT's inferred object-property fillers for (subj, prop) — the relational edges
    the reasoner entails (incl. those not directly asserted)."""
    node_set = onto.reasoner.owl_reasoner.getObjectPropertyValues(
        _ind(onto, subj_iri), _obj_prop(onto, prop_iri))
    return [str(i.getIRI()) for i in node_set.getFlattened()]


# ── inc-2d-full: corpus → ABox bridge (rows → individuals → realize → template labels) ──
ABOX_NS = "http://aegir.example.org/abox#"


def _san(s: str) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")


def template_iris(template) -> dict:
    """Deterministic ABox IRIs for one template: the subject class ``C_<tid>`` and, per
    object-property restriction ``{p} some {Y}``, a property + target class. Shared by the TBox
    generator and the row asserter so the asserted relation and the Domain/Range axiom agree."""
    from aegir.ontology.ddl import parse_restrictions
    tid = _san(template.template_id)
    rests = []
    for i, r in enumerate(parse_restrictions(template)):
        rests.append({"prop": f"{ABOX_NS}P_{tid}_{i}",
                      "target_class": f"{ABOX_NS}C_{tid}_{_san(r.target_slot)}",
                      "target_slot": r.target_slot})
    return {"template_id": template.template_id, "subject_class": f"{ABOX_NS}C_{tid}", "restrictions": rests}


def derive_domain_range_tbox(templates, *, with_domain_range: bool = True) -> str:
    """**The non-triviality step.** Lower each template's restrictions to a Manchester TBox: per
    template a subject class ``C_<tid>``; each ``{p} some {Y}`` → ``ObjectProperty P Domain C_<tid>
    Range C_<tid>_<Y>``. Asserting ``P(subject, obj)`` then lets HermiT *compute* ``subject : C_<tid>``
    (Domain) — recovering the template the row instantiates **without it being asserted**.

    ``with_domain_range=False`` is the **matched-token control**: identical classes + properties but
    NO Domain/Range, so the reasoner cannot infer a subject's type from its relations. The selectivity
    gap (full − control) IS the schema's contribution — capacity-free, byte-overlap-free.
    """
    lines = ["Ontology: <http://aegir.example.org/abox-corpus>"]
    classes: set[str] = set()
    props: list[str] = []
    for t in templates:
        info = template_iris(t)
        classes.add(info["subject_class"])
        for r in info["restrictions"]:
            classes.add(r["target_class"])
            if with_domain_range:
                props.append(f"ObjectProperty: <{r['prop']}> "
                             f"Domain: <{info['subject_class']}> Range: <{r['target_class']}>")
            else:
                props.append(f"ObjectProperty: <{r['prop']}>")
    for c in sorted(classes):
        lines.append(f"Class: <{c}>")
    return "\n".join(lines + props) + "\n"


def rows_to_individuals(onto, spine_table, rows, chapter_id: str, template) -> list[str]:
    """Assert one ABox individual per row + its object-property facts (FK columns) — **without**
    asserting the subject's type (that's what realization must compute). Returns the subject IRIs."""
    info = template_iris(template)
    cols = spine_table.table.columns
    pk_i = next((i for i, c in enumerate(cols) if c.slot_ref == "__pk__"), 0)
    rest_cols = []
    for r in info["restrictions"]:
        ci = next((i for i, c in enumerate(cols) if c.slot_ref == r["target_slot"]), None)
        if ci is not None:
            rest_cols.append((ci, r["prop"]))
    tid = _san(template.template_id)
    subj_iris: list[str] = []
    for ri, row in enumerate(rows):
        if pk_i >= len(row):
            continue
        subj = f"{ABOX_NS}i_{_san(chapter_id)}_{tid}_{ri}"
        subj_iris.append(subj)
        for ci, prop in rest_cols:
            if ci < len(row) and str(row[ci]).strip():
                obj = f"{ABOX_NS}o_{_san(chapter_id)}_{tid}_{ci}_{_san(row[ci])}"
                assert_relation(onto, subj, prop, obj)
    return subj_iris


def iris_to_label_idx(inferred_iris, template_index: dict[str, int]) -> list[int]:
    """Map HermiT-inferred subject-class IRIs (``…#C_<tid>``) back to template label indices.
    ``template_index`` is keyed by the SANITIZED template_id. Target/object classes (``C_<tid>_<Y>``)
    don't match a template key and are ignored."""
    out: set[int] = set()
    for iri in inferred_iris:
        local = re.split(r"[#/]", iri)[-1]
        if local.startswith("C_"):
            idx = template_index.get(local[2:])
            if idx is not None:
                out.add(idx)
    return sorted(out)


def realize_individuals(onto, ind_iris, direct: bool = False) -> dict[str, list[str]]:
    """Batch :func:`realize_types` over many individuals — call :func:`refresh` once first."""
    return {iri: realize_types(onto, iri, direct=direct) for iri in dict.fromkeys(ind_iris)}


def load_ontology(omn: str):
    """Write a Manchester ontology string to a temp .omn and load it under HermiT."""
    ensure_jvm()
    from deeponto.onto import Ontology
    with tempfile.NamedTemporaryFile(mode="w", suffix=".omn", delete=False) as fh:
        fh.write(omn)
        path = fh.name
    return Ontology(path, reasoner_type="hermit"), path


# Demo TBox: a property with Domain/Range. NO individual types are asserted in the TBox —
# the ABox (asserted below via OWLAPI) carries only the RELATION, so both endpoint types
# are computed by realization. This is the relational column-annotation G-rel was about.
_DEMO_TBOX = """Prefix: cco: <http://www.commoncoreontologies.org/>
Prefix: ex: <http://aegir.example.org/abox#>
Ontology: <http://aegir.example.org/abox-demo>
Class: cco:Artifact
Class: cco:InformationContentEntity
Class: ex:MedicinalHerbalPlant SubClassOf: cco:Artifact
Class: ex:BioactiveCompound SubClassOf: cco:InformationContentEntity
ObjectProperty: ex:hasActiveCompound Domain: ex:MedicinalHerbalPlant Range: ex:BioactiveCompound
"""

_EX = "http://aegir.example.org/abox#"


if __name__ == "__main__":
    onto, path = load_ontology(_DEMO_TBOX)
    plant, comp, prop = _EX + "plant1", _EX + "c1", _EX + "hasActiveCompound"
    MHP, BAC = _EX + "MedicinalHerbalPlant", _EX + "BioactiveCompound"

    # Assert ONLY the FK relation — neither endpoint's type is stated.
    assert_relation(onto, plant, prop, comp)
    refresh(onto)

    t_plant = realize_types(onto, plant, direct=True)
    t_comp = realize_types(onto, comp, direct=True)
    print(f"realized types(plant1) = {t_plant}")
    print(f"realized types(c1)     = {t_comp}")

    # Ground truth: the reasoner must COMPUTE both column types from the relation alone
    # (subject via Domain, object via Range) — neither was asserted.
    assert MHP in t_plant, f"subject column type not inferred from Domain: {t_plant}"
    assert BAC in t_comp, f"object/FK column type not inferred from Range: {t_comp}"
    Path(path).unlink(missing_ok=True)
    print("\nrealization-as-CPA verified: HermiT computed BOTH column types from the FK "
          "relation alone (Domain→subject, Range→object) — types nobody asserted. The "
          "relational/type skill (descoped G-rel) is computed by the reasoner, sound & complete.")
