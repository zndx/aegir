"""Closed generate → re-ground → refine engine — one episode against a FinePDFs cluster.

Control flow (skills_loop_spec.md §2.1, with DOF-walk amendments):
  seed (FinePDFs training cluster: target topic + evidence + ontology refs)
   → compose skills  S1 prose · S2 table · S3 diagram · S4 example · S6 links
                     (S7 conditions each toward the target; S5 grounds claims inline)
   → per-unit HARD gates + cheap deterministic repair  (r_axiom / claim_grounding / cross_modal)
   → assemble chapter
   → chapter-level HARD gate: topic_recovery ≥ τ_topic  (the expensive fidelity check)
   → admit iff  topic_recovery ≥ τ_topic  ∧  every unit passes its gates   (hard conjunction;
     the composite F is NOT consulted for admission — it ranks admitted units only).

Both ``generate_fn`` (the LLM backend) and ``topic_recovery_fn`` (BERTopic on the
cached FinePDFs model) are injected, so the whole episode is deterministically
testable (see ``scripts/smoke_engine.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegir.ontology.schema import Catalog
from aegir.ontology.skills.base import Gates, unit_gate_scores, unit_passes
from aegir.ontology.skills.library import (
    cross_reference,
    default_generate_fn,
    ground_claims,
    interleave_diagram,
    table_unit_from_s2,
    topic_anchor,
    verbalize_axiom,
    worked_example,
)
from aegir.ontology.skills.s2_relational_table import SourceSpan, synth_relational_table
from aegir.ontology.type_check import r_axiom


@dataclass
class Seed:
    cluster_id: str            # a FinePDFs *training* (non-holdout) BERTopic cluster
    target_topic: str          # the re-grounding coordinate
    refs: list[str]            # ontology template_ids to instantiate
    evidence: list[SourceSpan]


@dataclass
class EpisodeResult:
    units: list
    admitted: bool
    topic_recovery: float
    r_axiom: float             # the table unit's structure score
    per_unit_ok: bool
    rewards: dict = field(default_factory=dict)   # dense per-axis reward (MOPD-style); gates stay hard
    decisions: list[str] = field(default_factory=list)
    chapter_md: str = ""


def default_topic_recovery(chapter_md: str, target_topic: str) -> float:
    """Real backend = BERTopic re-recovery against the cached FinePDFs model
    (the calibration model at topic_recovery_v0/calibration). Inject a scorer for
    tests; bind the real one before the live run."""
    raise NotImplementedError(
        "default_topic_recovery: bind to the frozen FinePDFs BERTopic model, "
        "or inject topic_recovery_fn."
    )


def _regenerate(unit, seed: Seed, catalog: Catalog, generate_fn, gates: Gates, max_repair: int):
    """Re-run the unit's skill in repair mode (cheap deterministic checks only)."""
    k = unit.kind
    if k == "prose":
        return verbalize_axiom(seed.refs, seed.evidence, catalog, generate_fn, repair=True)
    if k == "diagram":
        return interleave_diagram(seed.refs, seed.evidence, catalog, generate_fn, repair=True)
    if k == "example":
        return worked_example(seed.refs, seed.evidence, catalog, generate_fn, repair=True)
    if k == "links":
        return cross_reference(seed.refs, seed.evidence, catalog, generate_fn, repair=True)
    if k == "table":
        res = synth_relational_table(
            seed.refs, seed.evidence, catalog,
            generate_fn=lambda p, r, e, repair=None: generate_fn("S2", seed.refs, seed.evidence, repair=True),
            tau_axiom=gates.tau_axiom, max_repair=max_repair)
        return table_unit_from_s2(res, seed.refs, seed.evidence)
    return unit


def run_episode(seed: Seed, catalog: Catalog,
                generate_fn=default_generate_fn,
                topic_recovery_fn=default_topic_recovery,
                gates: Gates | None = None, max_repair: int = 2,
                family_complex=None) -> EpisodeResult:
    if gates is None:
        gates = Gates()
    ev = seed.evidence
    ev_ids = {s.doc_id for s in ev}

    # S2 runs its own internal repair loop against r_axiom (its generate_fn adapter).
    s2res = synth_relational_table(
        seed.refs, ev, catalog,
        generate_fn=lambda p, r, e, repair=None: generate_fn("S2", seed.refs, ev, repair=bool(repair)),
        tau_axiom=gates.tau_axiom, max_repair=max_repair)

    units = [
        verbalize_axiom(seed.refs, ev, catalog, generate_fn),       # S1
        table_unit_from_s2(s2res, seed.refs, ev),                   # S2
        interleave_diagram(seed.refs, ev, catalog, generate_fn),    # S3
        worked_example(seed.refs, ev, catalog, generate_fn),        # S4
        cross_reference(seed.refs, ev, catalog, generate_fn, family_complex),  # S6
    ]
    # S7 condition + S5 ground (inline).
    units = [ground_claims(topic_anchor(u, seed.target_topic), ev) for u in units]

    # Per-unit hard gates + cheap repair.
    decisions: list[str] = []
    for i, u in enumerate(units):
        attempts = 0
        while not unit_passes(u, units, catalog, ev_ids, gates) and attempts < max_repair:
            u = ground_claims(topic_anchor(
                _regenerate(u, seed, catalog, generate_fn, gates, max_repair), seed.target_topic), ev)
            units[i] = u
            attempts += 1
        ok = unit_passes(u, units, catalog, ev_ids, gates)
        decisions.append(f"{u.provenance.skill_id}/{u.kind}: {'pass' if ok else 'FAIL'} "
                         f"({attempts} repair{'s' if attempts != 1 else ''})")

    # Chapter-level fidelity gate (the expensive check).
    chapter_md = "\n\n".join(u.content_md for u in units if u.content_md)
    tr = topic_recovery_fn(chapter_md, seed.target_topic)
    per_unit_ok = all(unit_passes(u, units, catalog, ev_ids, gates) for u in units)
    table_axiom = next((r_axiom(u.schema, catalog)[0] for u in units
                        if u.kind == "table" and u.schema), 0.0)

    # Dense per-axis reward (MOPD-style): the continuous gate scores exposed as a
    # reward vector for the calibration / on-policy phase. Admission stays the HARD
    # conjunction below — rewards never gate.
    axis: dict[str, list[float]] = {}
    for u in units:
        for k, v in unit_gate_scores(u, units, catalog, ev_ids).items():
            axis.setdefault(k, []).append(v)
    rewards = {k: sum(vs) / len(vs) for k, vs in axis.items()}
    rewards["topic_recovery"] = tr

    admitted = (tr >= gates.tau_topic) and per_unit_ok   # hard conjunction; F not consulted
    decisions.append(f"chapter topic_recovery={tr:.3f} (τ={gates.tau_topic}) → "
                     f"{'ADMIT' if admitted else 'REJECT'}")

    return EpisodeResult(units=units, admitted=admitted, topic_recovery=tr,
                         r_axiom=table_axiom, per_unit_ok=per_unit_ok, rewards=rewards,
                         decisions=decisions, chapter_md=chapter_md)
