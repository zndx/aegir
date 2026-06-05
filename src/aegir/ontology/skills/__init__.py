"""Skills library — reusable generative competencies for the closed
generate→re-ground→refine loop (see ``docs/current/src/ontology/skills_loop_spec.md``).

The library replaces the prescriptive template catalog as the generation *driver*:
the ontology supplies the *what*, skills supply the *how*, FinePDFs verifies *that
it held*. Each skill is a typed, versioned competency that must preserve the
re-grounding invariant (fidelity / structure / no-drift).
"""

from aegir.ontology.skills.s2_relational_table import (
    S2Result,
    SourceSpan,
    synth_relational_table,
)

__all__ = ["S2Result", "SourceSpan", "synth_relational_table"]
