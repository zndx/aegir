"""Skills S1, S3–S7 (S2 is in ``s2_relational_table.py``).

Each follows the S2 contract: grounded inputs (ontology refs + FinePDFs evidence)
→ a :class:`CorpusUnit` with provenance, with the LLM backend injected as a unified
``generate_fn(skill_id, refs, evidence, repair=False)`` so the skill logic is
deterministically testable. Payloads by skill:

  S1 verbalize-axiom    → (text, [Claim])             kind="prose"
  S3 interleave-diagram → (mermaid, [(src,dst)], [Claim])  kind="diagram"
  S4 worked-example     → (text, [Claim])             kind="example"
  S6 cross-reference    → (text, [Claim])             kind="links"  (family-complex gated)
  S7 topic-anchor       → conditioner (tags target_topic; effect read by chapter topic_recovery)
  S5 ground-claim       → verifier (attaches grounded_to; the no-drift check)

See ``docs/current/src/ontology/skills_loop_spec.md`` §1.
"""

from __future__ import annotations

from aegir.ontology.schema import Catalog
from aegir.ontology.skills.base import CorpusUnit, Provenance
from aegir.ontology.skills.s2_relational_table import S2Result, SourceSpan


def _ev_ids(evidence: list[SourceSpan]) -> list[str]:
    return [s.doc_id for s in evidence]


def default_generate_fn(skill_id: str, refs, evidence, repair: bool = False):
    """Unified backend stub — bind to the GLM/Grok mix before the live run; inject
    a deterministic ``generate_fn`` for tests/standalone use (see scripts/smoke_engine.py)."""
    raise NotImplementedError(
        f"default_generate_fn[{skill_id}]: bind to the generation mix or inject generate_fn."
    )


# ── Generators ───────────────────────────────────────────────────────────────


def verbalize_axiom(refs, evidence, catalog: Catalog, generate_fn, repair=False) -> CorpusUnit:  # S1
    text, claims = generate_fn("S1", refs, evidence, repair=repair)
    return CorpusUnit("prose", text, Provenance("S1", list(refs), _ev_ids(evidence), list(claims)))


def interleave_diagram(refs, evidence, catalog: Catalog, generate_fn, repair=False) -> CorpusUnit:  # S3
    mermaid, edges, claims = generate_fn("S3", refs, evidence, repair=repair)
    return CorpusUnit("diagram", mermaid,
                      Provenance("S3", list(refs), _ev_ids(evidence), list(claims)),
                      diagram_edges=list(edges))


def worked_example(refs, evidence, catalog: Catalog, generate_fn, repair=False) -> CorpusUnit:  # S4
    text, claims = generate_fn("S4", refs, evidence, repair=repair)
    return CorpusUnit("example", text, Provenance("S4", list(refs), _ev_ids(evidence), list(claims)))


def cross_reference(refs, evidence, catalog: Catalog, generate_fn,
                    family_complex=None, repair=False) -> CorpusUnit:  # S6
    """Links concepts; family-complex-gated so only co-coherent family-sets are linked
    (drop offending families to the largest allowed face, never a measured puncture)."""
    cited = list(refs)
    if family_complex is not None:
        families = _families_of(refs, catalog)
        if hasattr(family_complex, "is_allowed") and not family_complex.is_allowed(families):
            allowed = family_complex.best_face(families) if hasattr(family_complex, "best_face") else families
            cited = [r for r in refs if _family_of(r, catalog) in set(allowed)]
    text, claims = generate_fn("S6", cited, evidence, repair=repair)
    return CorpusUnit("links", text, Provenance("S6", cited, _ev_ids(evidence), list(claims)))


# ── Conditioner (S7) + verifier (S5) ─────────────────────────────────────────


def topic_anchor(unit: CorpusUnit, target_topic: str) -> CorpusUnit:  # S7
    """Tag the unit with its FinePDFs target topic; the re-grounding effect is read
    by the chapter-level topic_recovery gate."""
    unit.provenance.target_topic = target_topic
    return unit


def ground_claims(unit: CorpusUnit, evidence: list[SourceSpan]) -> CorpusUnit:  # S5
    """Attach ``grounded_to`` to each claim that a FinePDFs evidence span supports
    (substring overlap, deterministic); unsupported claims stay ungrounded and will
    fail the no-drift gate."""
    ev = {s.doc_id: (s.text or "").lower() for s in evidence}
    for c in unit.provenance.claims:
        if c.grounded_to is not None:
            continue
        ct = (c.text or "").lower()
        for did, t in ev.items():
            if ct and (ct in t or t in ct):
                c.grounded_to = did
                break
    return unit


# ── S2 → CorpusUnit adapter ──────────────────────────────────────────────────


def table_unit_from_s2(result: S2Result, refs, evidence: list[SourceSpan]) -> CorpusUnit:
    md = _render_tables_md(result.schema)
    return CorpusUnit("table", md,
                      Provenance("S2", list(refs), _ev_ids(evidence), claims=[]),
                      schema=result.schema)


# ── helpers ──────────────────────────────────────────────────────────────────


def _family_of(template_id: str, catalog: Catalog) -> str | None:
    try:
        prov = catalog.by_id(template_id).provenance or {}
        return prov.get("family")
    except KeyError:
        return None


def _families_of(refs, catalog: Catalog) -> list[str]:
    return sorted({f for f in (_family_of(r, catalog) for r in refs) if f})


def _render_tables_md(schema) -> str:
    out = []
    for t in schema.tables:
        head = " | ".join(c.name for c in t.columns)
        sep = " | ".join("---" for _ in t.columns)
        rows = "\n".join(" | ".join(r) for r in t.rows)
        out.append(f"**{t.name}**\n\n{head}\n{sep}\n{rows}")
    return "\n\n".join(out)
