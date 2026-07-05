"""Metrology-informed derivation loop (#141) — the agent RESPONDS to the SchemaPile signal.

The agent-mediated feedback loop doctrine applied to structure: ``derive_harness.propose`` gives
entity records; ``ddl_membrane.structural_signal`` profiles the DDL they yield against the
empirical SchemaPile norms and returns a verdict + an agent-facing ``reason``. Below ``rich``,
the reason goes BACK to the agent as re-prompt context (submit → gate → re-prompt-with-failure →
resubmit) — never a silent drop. The membrane is deterministic (kvasir, ms); the loop is bounded.

This is what "agents informed by SchemaPile metrology" means concretely: the FinePDFs→ontology
agent iterates against the structural ground truth until its schema is real-world-shaped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aegir.ontology.derive_harness import propose
from aegir.ontology.entities import Entity, to_manchester

_RANK = {"malformed": 0, "inert": 1, "thin": 2, "rich": 3}


@dataclass
class DeriveReport:
    """Trace of one metrology-informed derivation (persisted to the run manifest)."""
    rounds: list[dict] = field(default_factory=list)
    final_verdict: str = ""
    accepted_round: int = 0

    def summary(self) -> dict:
        return {"rounds": self.rounds, "final_verdict": self.final_verdict,
                "accepted_round": self.accepted_round}


def _feedback(signal: dict) -> str:  # noqa: D103 — see _dormant_brief below
    """The membrane's reason, framed as re-prompt context for the next proposal round."""
    y = signal.get("ddl_yield") or {}
    return (
        "Your previous schema draft was profiled against real-world database norms "
        f"(verdict: {signal.get('verdict')}). Measured: {y.get('n_elected', 0)} entity tables, "
        f"{y.get('n_junctions', 0)} junction tables, {y.get('n_lookups', 0)} lookup tables, "
        f"attribute-less-class ratio {y.get('attr_zero_ratio', 1.0):.2f}, median width "
        f"{y.get('width_median', 0)}. {signal.get('reason', '')} "
        "Revise: keep the same subject-matter entities but deepen them — more typed attributes "
        "per entity (dates, quantities, codes, booleans), closed value sets (enum) where the "
        "passage implies categories, and cardinality-bounded relations between entities. "
        "Model the passage's OWN domain; do not invent unrelated entities."
        + _dormant_brief(signal)
    )


_DORMANT_HINTS = {
    "k2_inverse_functional": "an IDENTITY-bearing relation (a badge/license/account that "
                             "uniquely identifies its bearer — mark it identifying)",
    "k4_oneof_enum": "a CLOSED enumeration the domain fixes (statuses, grades, phases) — "
                     "as an attribute with an enum value set",
    "oneof_class": "a closed category kind",
    "junction_tables": "a genuine many-to-many association (min/max cardinality > 1)",
    "lookup_tables": "closed value sets (enum attributes)",
    "pk_natural": "a domain identifier attribute (accession number, code, serial)",
    "composite_key_deferred": "a weak entity identified by its parent plus a local ordinal",
    "cardinality_many": "bounded multiplicity (min 2 / max N) where the domain implies it",
}


def _dormant_brief(signal: dict) -> str:
    dormant = [d for d in (signal.get("dormant_paths") or []) if d in _DORMANT_HINTS]
    if not dormant:
        return ""
    wants = "; ".join(_DORMANT_HINTS[d] for d in dormant[:4])
    return (" Structural coverage note: this draft exercises none of the following real-world "
            f"schema patterns — {wants}. If the domain PLAUSIBLY carries any of them, model "
            "them; interesting structure is welcome where the subject matter supports it.")


def derive_with_metrology(passage: str, *, max_rounds: int = 2, temperature: float = 0.3,
                          accept: str = "rich") -> "tuple[list[Entity], DeriveReport]":
    """Propose → profile vs SchemaPile → re-prompt with the structural reason → accept best.

    Keeps the best round by verdict rank (never regresses on a worse revision). ``accept``
    is the early-exit verdict; ``max_rounds`` bounds the loop (deterministic membrane, ms per
    check — the engine call is the cost)."""
    from aegir.ontology.ddl_membrane import structural_signal

    report = DeriveReport()
    best: "tuple[int, list[Entity]]" = (-1, [])
    context = ""
    for rnd in range(1, max_rounds + 1):
        entities, meta = propose(passage, temperature=temperature, context=context)
        signal = structural_signal(to_manchester(entities)) if entities else {
            "verdict": "malformed", "reason": "no entities parsed"}
        rank = _RANK.get(signal.get("verdict", "malformed"), 0)
        report.rounds.append({
            "round": rnd, "n_entities": len(entities), "verdict": signal.get("verdict"),
            "ddl_yield": signal.get("ddl_yield"), "parity": signal.get("parity"),
            "reasoning_len": len(meta.get("reasoning", "") or "")})
        if rank > best[0]:
            best = (rank, entities)
            report.accepted_round = rnd
        if signal.get("verdict") == accept:
            break
        context = _feedback(signal)
    report.final_verdict = next(
        (r["verdict"] for r in report.rounds if r["round"] == report.accepted_round), "")
    return best[1], report


def merge_entities(entity_sets: "list[list[Entity]]") -> "list[Entity]":
    """Union per-passage entity sets into one ontology: same-name entities merge (attributes
    by name, relations by (prop, target)); first-seen genus/definition wins. Cross-passage
    relations whose target class exists anywhere in the union survive — the cross-domain
    edges emerge from the union rather than being injected."""
    merged: "dict[str, Entity]" = {}
    for ents in entity_sets:
        for e in ents:
            key = e.iri()
            if key not in merged:
                merged[key] = Entity(name=e.name, label=e.label, genus=e.genus,
                                     definition=e.definition,
                                     attributes=list(e.attributes), relations=list(e.relations))
                continue
            m = merged[key]
            have_a = {a.name for a in m.attributes}
            m.attributes.extend(a for a in e.attributes if a.name not in have_a)
            have_r = {(r.prop, r.target) for r in m.relations}
            m.relations.extend(r for r in e.relations if (r.prop, r.target) not in have_r)
    out = list(merged.values())
    known = {e.iri() for e in out}
    for e in out:  # drop dangling relations (target never defined anywhere)
        e.relations = [r for r in e.relations if r.target_iri() in known]
    return out
