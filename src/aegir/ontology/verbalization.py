"""Multi-frame verbalization renderer (Semantic-Layer-Upkeep Comp 3).

Replaces the single flat ``verbal_template`` ("{X} is something that {p} {Y}", the 167×-repeated
"§ is a · that · §" frame) with a SET of diverse, *procedural*, slot-faithful surface forms per template.

Two layers, separated so the diversity engine is testable without the JVM:

  * ``compose_frames(parts, ...)`` — **pure**: turns extracted semantic parts (subject, quantified
    constraints, named supers) into N surface frames spanning genuinely different syntactic skeletons
    (subsumptive / definitional / necessary-condition / relational-fronted / normative). Slot-faithful by
    construction (frames only add fixed scaffolding around the slot-carrying parts) and verified by
    ``frames_preserve_slots``. A **seeded per-template subset** is emitted so the global skeleton
    distribution FLATTENS (uniform frames would instead concentrate top-5 share) — see compose_frames.
  * ``extract_parts(template)`` — **JVM-backed** (DeepOnto): walks the ``OntologyVerbaliser`` CfgNode parse
    tree (NOT its single output string — "DeepOnto is not a black box") into ``VerbalizationParts``. Lives
    in ``deeponto_harness`` alongside the render/load machinery; re-exported here.

The deterministic frames raise distinct-skeleton count; LLM elaboration (scripts/elaborate_verbalizations.py,
via the local engine) adds natural-language variety that flattens top-5 share further. Both are scored by
``audit_verbalization_entropy`` and gated by ``semantic_layer_gate``.

PROVISIONAL (provisional_scaffolding_not_goals): templated frames are scaffolding to lift verbalization
diversity off the floor; the north star is genuinely rich procedural prose grounded in the ontology.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SLOT_RE = re.compile(r"\{[^}]*\}")


@dataclass
class Constraint:
    """One quantified relational restriction: ``property quantifier filler`` (e.g. ``{p} some {Y}``)."""
    property: str          # verb/relation phrase, slot-carrying (e.g. "{p}") or a concrete label
    filler: str            # the filler noun phrase, slot-carrying (e.g. "{Y}") or a concrete label
    quantifier: str = "some"   # "some" | "only" | "exactly" | "min" | "max" | ""
    negated: bool = False
    count: "int | None" = None   # cardinality for exactly/min/max (recovered from Manchester —
    #                              DeepOnto's cardinality support is incomplete/raising, and its
    #                              exceptions previously dropped these restrictions entirely)


@dataclass
class VerbalizationParts:
    """The semantic decomposition of a template's defining axiom (markers already lowered to ``{slot}``)."""
    subject: str                                  # anchor noun phrase, slot-carrying (e.g. "{X}")
    constraints: list[Constraint] = field(default_factory=list)
    named_supers: list[str] = field(default_factory=list)   # named parents (e.g. ["artifact"])
    axiom_kind: str = "subsumption"               # "subsumption" | "equivalence" | "assertion"

    @property
    def is_relational(self) -> bool:
        return any(c.property for c in self.constraints)


# ── grammar helpers ───────────────────────────────────────────────────────────

_VOWEL = re.compile(r"^[aeiouAEIOU]")


def _art(noun: str) -> str:
    """Indefinite article for a phrase; slot phrases ('{Y}') get 'a' (filled later)."""
    core = noun.strip().lstrip("{").rstrip("}")
    return "an" if _VOWEL.match(core) else "a"


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine"}


def _numword(n: "int | None") -> str:
    return _NUMWORD.get(n or 1, str(n))


def _count(c: "Constraint | str") -> str:
    """Natural count phrase for a constraint (cardinality-aware). Accepts a bare quantifier string
    for back-compat with pre-cardinality callers."""
    q = c if isinstance(c, str) else c.quantifier
    n = None if isinstance(c, str) else c.count
    if q == "exactly":
        return f"exactly {_numword(n)}"
    if q == "min":
        return f"at least {_numword(n)}"
    if q == "max":
        return f"at most {_numword(n)}"
    return {"some": "at least one", "only": "only", "": "a"}.get(q, "a")


def _qword(c: "Constraint | str") -> str:
    q = c if isinstance(c, str) else c.quantifier
    if q in ("exactly", "min", "max"):
        return _count(c)                      # cardinalities read as their count phrase in verb frames
    return {"some": "some", "only": "only", "": ""}.get(q, "")


def _head(parts: VerbalizationParts, default: str) -> str:
    """The head noun phrase: the named supers if any, else a generic fallback ('entity'/'thing')."""
    return " ".join(parts.named_supers) if parts.named_supers else default


def slots_of(text: str) -> set[str]:
    return set(_SLOT_RE.findall(text))


# ── clause renderers (one constraint → a clause; robust to {p} being an unknown verb/relation) ──

def _verb_rel(c: Constraint, connective: str = "that") -> str:
    """Verb-framing as DeepOnto uses it: '<conn> {p} some {Y}'. Reads given {p} fills as a verb."""
    neg = "does not " if c.negated else ""
    q = _qword(c)
    return " ".join(w for w in (connective, neg + c.property, q, c.filler) if w).strip()


def _rel_phrase(c: Constraint) -> str:
    """Relation-noun phrase WITHOUT a verb, so a single fronted verb can govern a list of them (avoids
    'stands ... and stand ...'): 'the {p} relation to at least one {Y}'. Robust to {p}'s filled form."""
    if c.negated:
        return f"no {c.property} relation to any {c.filler}"
    return f"the {c.property} relation to {_count(c)} {c.filler}"


def _via(c: Constraint) -> str:
    """'to at least one {Y} via {p}' — relational-fronted tail."""
    if c.negated:
        return f"to no {c.filler} via {c.property}"
    return f"to {_count(c)} {c.filler} via {c.property}"


def _join(clauses: list[str], conj: str = "and") -> str:
    clauses = [c for c in clauses if c]
    if len(clauses) <= 1:
        return clauses[0] if clauses else ""
    return ", ".join(clauses[:-1]) + f", {conj} " + clauses[-1]


# ── Manchester-side recovery (RH 2026-07-19: DeepOnto cardinality is incomplete/raising) ──────

_REL_VERB = {  # curated BFO/CCO-ish relation → verb phrase (de-camel fallback handles the rest)
    "realizes": "realizes", "realizedIn": "is realized in", "borneBy": "is borne by",
    "bears": "bears", "hasBearer": "has bearer", "inheresIn": "inheres in",
    "participatesIn": "participates in", "hasParticipant": "has participant",
    "isAbout": "is about", "designates": "designates", "prescribes": "prescribes",
    "hasPart": "has part", "partOf": "is part of",
}
_MREST = re.compile(r"(?:[\w-]+:)?(\w+)\s+(some|only|exactly|min|max)\s*(\d+)?\s+\{(\w+):Class\}")


def _prop_phrase(local: str) -> str:
    if local in _REL_VERB:
        return _REL_VERB[local]
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", local).lower()
    return f"is {words}" if words.endswith(" by") else words


def merge_manchester_restrictions(parts: VerbalizationParts, manchester: str) -> VerbalizationParts:
    """Recover what DeepOnto dropped or degraded, from the axiom's own Manchester source (authoritative).

    DeepOnto's verbaliser raises on cardinality restrictions (exactly/min/max N) — the harness's
    per-expression ``except: continue`` then lost the WHOLE restriction, and 'exactly one' phrasings
    were lucky outcomes at best (RH). This pure pass re-reads the template's Manchester: an existing
    walked constraint on the same filler gets its degraded quantifier UPGRADED; a restriction absent
    from the walk is APPENDED (slot-faithful '{Slot}' filler; property via the curated lexicon with
    de-camel fallback, participles prefixed 'is')."""
    for m in _MREST.finditer(manchester or ""):
        prop, quant, num, slot = m.group(1), m.group(2), m.group(3), m.group(4)
        count = int(num) if num else None
        existing = next((c for c in parts.constraints if slot in c.filler), None)
        if existing is not None:
            if quant in ("exactly", "min", "max") and existing.quantifier not in ("exactly", "min", "max"):
                existing.quantifier, existing.count = quant, count
            # normalize DeepOnto's bare participles ('borne by' → 'is borne by') so verb frames read
            if existing.property.endswith(" by") and not existing.property.startswith(("is ", "are ")):
                existing.property = "is " + existing.property
            continue
        parts.constraints.append(Constraint(property=_prop_phrase(prop), filler="{" + slot + "}",
                                            quantifier=quant, count=count))
    return parts


# ── frame composition ─────────────────────────────────────────────────────────

def _relational_frames(p: VerbalizationParts) -> list[str]:
    S, C = p.subject, p.constraints
    head = _head(p, "entity")
    art_head = f"{_art(head)} {head}"
    is_equiv = p.axiom_kind == "equivalence"
    frames = [
        # subsumptive (verb-framing; matches the legacy form for continuity)
        f"{S} is {art_head} {_join([_verb_rel(c, 'that') for c in C])}".strip(),
        # definitional (verb-framing, 'which' → distinct skeleton; 'Each' avoids an unknown-article '{S}')
        f"Each {S} is defined as {art_head} {_join([_verb_rel(c, 'which') for c in C])}.",
        # necessary-condition (procedural; one fronted verb governs the relation-phrase list)
        f"For {_art(head)} {head} to count as {S}, it must bear {_join([_rel_phrase(c) for c in C])}.",
        # normative (every-stands; single fronted 'stands in')
        f"Every {S} stands in {_join([_rel_phrase(c) for c in C])}.",
    ]
    # BFO-aware role framing (RH): roles read best with their bearer/realization fronted
    if "role" in head.lower() and C:
        frames.append(f"As a role, {S} is characterised by {_join([_rel_phrase(c) for c in C])}.")
    if is_equiv:
        frames.append(f"{S} is exactly {art_head} {_join([_verb_rel(c, 'that') for c in C])}.")
    return frames


def _bare_frames(p: VerbalizationParts) -> list[str]:
    S = p.subject
    sup = _head(p, "entity")
    art_sup = f"{_art(sup)} {sup}"
    return [
        f"{S} is a kind of {sup}.",
        f"{S} is a type of {sup}.",
        f"Every {S} is {art_sup}.",
    ]


def compose_frames(parts: VerbalizationParts, *, seed_key: str = "", k: int = 5) -> list[str]:
    """Compose up to ``k`` diverse, slot-faithful surface frames for ``parts``.

    A **seeded subset** (keyed by ``seed_key``, typically the template_id) is chosen from the applicable
    pool so different templates contribute different skeleton mixes — flattening the global distribution
    (emitting the same frames for every template would concentrate top-5 share instead of spreading it).
    The first frame (subsumptive) is always kept for continuity; the rest are seeded-shuffled. Frames that
    fail slot-preservation are dropped.
    """
    pool = _relational_frames(parts) if parts.is_relational else _bare_frames(parts)
    pool = [re.sub(r"\s+", " ", f).strip() for f in pool]
    want = slots_of(" ".join([parts.subject] + parts.named_supers
                             + [c.property for c in parts.constraints]
                             + [c.filler for c in parts.constraints]))
    # keep only frames that preserve exactly the slot set (no drops, no spurious slots)
    pool = [f for f in pool if slots_of(f) == want]
    # dedupe preserving order
    seen: set[str] = set()
    pool = [f for f in pool if not (f in seen or seen.add(f))]
    if not pool:
        return []
    head, tail = pool[0], pool[1:]
    rng = _seeded(seed_key or parts.subject)
    rng.shuffle(tail)
    out = [head] + tail
    return out[:k]


def _seeded(key: str):
    import random
    h = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(h)


def frames_preserve_slots(frames: list[str], expected: set[str]) -> bool:
    """True iff every frame carries exactly ``expected`` slot set — the slot-faithfulness invariant."""
    return all(slots_of(f) == expected for f in frames)


# ── extract_parts is JVM-backed; re-exported from the harness to keep this module importable JVM-free ──

def extract_parts(template, *, _harness=None) -> VerbalizationParts | None:
    """Walk a template's DeepOnto CfgNode parse tree into VerbalizationParts (JVM required).

    Thin re-export: the implementation lives in ``deeponto_harness.extract_parts`` (it needs the render/
    load machinery + an active JVM). Imported lazily so this module stays JVM-free for the pure composer.
    """
    from aegir.ontology import deeponto_harness as h
    return h.extract_parts(template)
