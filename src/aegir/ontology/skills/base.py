"""Shared skill-layer types + deterministic gate checkers for the closed loop.

``CorpusUnit`` is the common output of every skill. The re-grounding invariant is
enforced as **hard, per-modality gates** (DOF-walk amendment 1): a unit is admitted
only if every gate applicable to its modality passes — the composite ``F`` (deferred)
ranks admitted units but never gates. ``topic_recovery`` is a *chapter-level* gate
(amendment 3) and lives in the engine; the per-unit gates live here.

Per-modality applicability (DOF 5 generalized):
  table   → r_axiom + claim_grounding      diagram → cross_modal
  prose / example / links → claim_grounding

See ``docs/current/src/ontology/skills_loop_spec.md`` §2 and the DOF review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegir.ontology.schema import Catalog
from aegir.ontology.type_check import UnitSchema, r_axiom


@dataclass
class Claim:
    text: str
    grounded_to: str | None = None   # source_span_id / axiom id; None = ungrounded (drift)


@dataclass
class Provenance:
    skill_id: str
    ontology_refs: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    target_topic: str | None = None


@dataclass
class CorpusUnit:
    kind: str                                       # prose | table | diagram | example | links
    content_md: str
    provenance: Provenance
    schema: UnitSchema | None = None                # tables only
    diagram_edges: list[tuple[str, str]] | None = None  # diagrams only


@dataclass
class Gates:
    tau_topic: float = 0.80     # chapter-level fidelity (BERTopic re-grounding)
    tau_axiom: float = 0.45     # structure (relational units)
    tau_ground: float = 0.95    # no-drift (claim grounding)
    tau_x: float = 0.90         # cross-modal consistency (diagrams)


# Hard admission gates = the re-grounding INVARIANT components only (structure =
# r_axiom, no-drift = claim_grounding; fidelity = topic_recovery is chapter-level).
# cross_modal (diagram↔table consistency) is a QUALITY signal, NOT an invariant
# component, so it is reward-only (ranking) rather than a hard gate — per the
# DOF-walk gate/rank split. (Refinement to amendment #3, motivated by the pilot:
# diagram flakiness must not reject chapters that pass fidelity/structure/no-drift.)
GATES_BY_KIND: dict[str, tuple[str, ...]] = {
    "table": ("r_axiom", "claim_grounding"),
    "prose": ("claim_grounding",),
    "example": ("claim_grounding",),
    "links": ("claim_grounding",),
    "diagram": (),
}


def claim_grounding(unit: CorpusUnit, evidence_span_ids: set[str]) -> float:
    """Fraction of a unit's claims grounded to an evidence span (no-drift). A unit
    that asserts nothing is vacuously grounded (1.0)."""
    claims = unit.provenance.claims
    if not claims:
        return 1.0
    ok = sum(1 for c in claims if c.grounded_to is not None and c.grounded_to in evidence_span_ids)
    return ok / len(claims)


def cross_modal(unit: CorpusUnit, chapter: list[CorpusUnit]) -> float:
    """A diagram's edges must correspond to a foreign-key edge in some table unit
    (the diagram cannot introduce relations the relational data doesn't carry)."""
    if unit.kind != "diagram" or not unit.diagram_edges:
        return 1.0
    fk_pairs: set[tuple[str, str]] = set()
    for u in chapter:
        if u.schema:
            for e in u.schema.fk_edges:
                fk_pairs.add((e.src_col, e.dst_col))
    ok = sum(1 for a, b in unit.diagram_edges if (a, b) in fk_pairs or (b, a) in fk_pairs)
    return ok / len(unit.diagram_edges)


def unit_gate_scores(unit: CorpusUnit, chapter: list[CorpusUnit],
                     catalog: Catalog, evidence_span_ids: set[str]) -> dict[str, float]:
    """Per-unit scores for exactly the gates applicable to the unit's modality."""
    scores: dict[str, float] = {}
    for gate in GATES_BY_KIND.get(unit.kind, ("claim_grounding",)):
        if gate == "r_axiom":
            scores["r_axiom"] = r_axiom(unit.schema, catalog)[0] if unit.schema else 0.0
        elif gate == "claim_grounding":
            scores["claim_grounding"] = claim_grounding(unit, evidence_span_ids)
        elif gate == "cross_modal":
            scores["cross_modal"] = cross_modal(unit, chapter)
    return scores


def unit_passes(unit: CorpusUnit, chapter: list[CorpusUnit], catalog: Catalog,
                evidence_span_ids: set[str], gates: Gates) -> bool:
    """Hard per-unit gate: ALL applicable gates must clear their thresholds (no weighting)."""
    thr = {"r_axiom": gates.tau_axiom, "claim_grounding": gates.tau_ground,
           "cross_modal": gates.tau_x}
    return all(v >= thr[k] for k, v in
               unit_gate_scores(unit, chapter, catalog, evidence_span_ids).items())
