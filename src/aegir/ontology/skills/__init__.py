"""Skills library — reusable generative competencies for the closed
generate→re-ground→refine loop (see ``docs/current/src/ontology/skills_loop_spec.md``).

The library replaces the prescriptive template catalog as the generation *driver*:
the ontology supplies the *what*, skills supply the *how*, FinePDFs verifies *that
it held*. Each skill is a typed, versioned competency that must preserve the
re-grounding invariant (fidelity / structure / no-drift), enforced as hard
per-modality gates (see ``base``) by the engine (``aegir.ontology.engine``).
"""

from aegir.ontology.skills.base import (
    Claim,
    CorpusUnit,
    Gates,
    Provenance,
    claim_grounding,
    cross_modal,
    unit_passes,
)
from aegir.ontology.skills.library import (
    cross_reference,        # S6
    ground_claims,          # S5
    interleave_diagram,     # S3
    table_unit_from_s2,
    topic_anchor,           # S7
    verbalize_axiom,        # S1
    worked_example,         # S4
)
from aegir.ontology.skills.s2_relational_table import (
    S2Result,
    SourceSpan,
    synth_relational_table,  # S2
)

__all__ = [
    # base types + gate checkers
    "Claim", "CorpusUnit", "Gates", "Provenance",
    "claim_grounding", "cross_modal", "unit_passes",
    # skills S1–S7
    "verbalize_axiom", "synth_relational_table", "interleave_diagram",
    "worked_example", "ground_claims", "cross_reference", "topic_anchor",
    "table_unit_from_s2",
    "S2Result", "SourceSpan",
]
