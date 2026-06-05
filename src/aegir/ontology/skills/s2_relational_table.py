"""S2 · synth-relational-table-with-cross-FKs — the label-bearing core skill.

Given ontology template refs + FinePDFs evidence spans, generate relational
table(s) that instantiate the templates' slot structure: Class/Data slots become
type-correct columns (each column carrying its ``slot_ref`` — the deterministic
CTA/CPA/DED label), ObjectProperty slots become foreign-key edges, and cell values
are grounded in the evidence. The skill runs the **per-unit iterative repair loop**
(DOF 6): generate → ``r_axiom`` type-check → regenerate only the offending
columns/edges, up to ``max_repair`` attempts, before admission at ``r_axiom ≥
tau_axiom``.

Grounding granularity (DOF 5): column→ontology (slot_ref/slot_type, deterministic),
table→FinePDFs (cell-value distribution re-grounds — verified by the engine's
BERTopic step, not here), row→evidence (each populated row traces to ≥1 SourceSpan).

The LLM call is injected as ``generate_fn`` so the skill logic (prompt construction,
parse, type-check, repair) is testable deterministically. The default binds to the
GLM/Grok generation mix; tests/standalone callers pass their own ``generate_fn``.

See ``docs/current/src/ontology/skills_loop_spec.md`` §1 (S2) and §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aegir.ontology.schema import Catalog
from aegir.ontology.type_check import AxiomBreakdown, UnitSchema, r_axiom


@dataclass
class SourceSpan:
    """A FinePDFs evidence span — the re-grounding anchor (the *ground*)."""
    doc_id: str
    text: str
    char_range: tuple[int, int] | None = None
    topic_vec: list[float] | None = None


# A generation backend: (prompt, refs, evidence, repair?) -> UnitSchema.
GenerateFn = Callable[..., UnitSchema]


@dataclass
class S2Result:
    schema: UnitSchema
    r_axiom: float
    breakdown: AxiomBreakdown
    attempts: int          # number of *repair* attempts (0 = admitted on first pass)
    admitted: bool
    history: list[float] = field(default_factory=list)  # r_axiom per attempt


def build_prompt(refs: list[str], evidence: list[SourceSpan]) -> str:
    """The S2 generation prompt skeleton (spec §1, S2c)."""
    ev = "\n".join(f"- [{s.doc_id}] {s.text}" for s in evidence) or "(no evidence)"
    return (
        "Instantiate the following ontology templates as relational tables.\n"
        "Rules:\n"
        "  • one table per template; its columns are the template's Class/Data slots;\n"
        "  • populate cells with realistic values GROUNDED in the evidence below;\n"
        "  • where an object-property slot relates two classes, emit a foreign-key edge;\n"
        "  • output (1) markdown tables and (2) a JSON schema giving, for every column,\n"
        "    its ontology slot_ref + slot_type, and the fk_edges (with via_slot).\n\n"
        f"Templates: {', '.join(refs)}\n\n"
        f"Evidence (FinePDFs spans — the values must re-ground here):\n{ev}\n"
    )


def build_repair_prompt(refs: list[str], evidence: list[SourceSpan],
                        schema: UnitSchema, offending) -> str:
    """Targeted repair prompt — regenerate only the offending slots/edges."""
    problems = "\n".join(
        f"  • table {c.table!r} slot {c.slot!r} ({c.owl_type}): {c.reason}"
        for c in offending
    )
    return (
        build_prompt(refs, evidence)
        + "\nThe previous attempt FAILED the structural type-check on these slots; "
        "fix ONLY these, leaving the rest unchanged:\n" + problems + "\n"
    )


def default_generate_fn(prompt: str, refs: list[str],
                        evidence: list[SourceSpan], repair=None) -> UnitSchema:
    """Default backend — binds to the GLM/Grok mix used by ``generate_chapter``.

    Intentionally not wired in this commit: the skill logic (prompt/parse/type-check/
    repair) is what S2 contributes and what the smoke validates; binding the LLM mix
    is the next integration step before the *live* generation run. Inject a
    ``generate_fn`` for tests and standalone use.
    """
    raise NotImplementedError(
        "default_generate_fn: bind to the GLM/Grok mix (reuse scripts/generate_chapter "
        "machinery) before the live run, or pass an explicit generate_fn."
    )


def synth_relational_table(
    refs: list[str],
    evidence: list[SourceSpan],
    catalog: Catalog,
    generate_fn: GenerateFn = default_generate_fn,
    tau_axiom: float = 0.45,
    max_repair: int = 2,
) -> S2Result:
    """Generate a relational unit and repair it to ``r_axiom ≥ tau_axiom``.

    Args:
        refs: template_ids to instantiate.
        evidence: FinePDFs SourceSpans grounding the values.
        catalog: the ontology catalog (for the type-check).
        generate_fn: backend producing a :class:`UnitSchema` (LLM by default).
        tau_axiom: admission threshold (spec §2.3 default 0.45).
        max_repair: max repair attempts after the first generation.
    """
    schema = generate_fn(build_prompt(refs, evidence), refs, evidence)
    score, bd = r_axiom(schema, catalog)
    history = [score]
    attempts = 0

    while score < tau_axiom and attempts < max_repair:
        schema = generate_fn(
            build_repair_prompt(refs, evidence, schema, bd.offending()),
            refs, evidence, repair=bd.offending(),
        )
        score, bd = r_axiom(schema, catalog)
        history.append(score)
        attempts += 1

    return S2Result(
        schema=schema, r_axiom=score, breakdown=bd,
        attempts=attempts, admitted=score >= tau_axiom, history=history,
    )
