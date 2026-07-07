"""The annotation membrane — deterministic gates for AUTHORED SKOS surfaces (4b).

The inverted topic layer's definitional-rigor loop has an authored half: an agent
proposes retrieval surfaces (altLabels · scopeNote · an expanded definition) for terms
whose declared surface is too thin to discriminate. Annotations bypass HermiT (no OWL
axioms change), so the membranes here are the ONLY gates — all deterministic, and every
verdict returns its REASON (the aegir closed-loop doctrine: the agent responds to the
gate, so the gate must speak).

M1 shape · M2 token bounds vs the registry's granularity · M3 no smuggled ontological
claims · M4 vocabulary grounding in the term's declared/allowed set · M5 clean-room
tripwire · M6 discrimination (dedup + the self-retrieval margin gate — the increment's
objective, measured at authoring time).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from aegir.ontology.patterns import _FORBIDDEN

# stricter numeric-code variant (SNOMED/LOINC leak into prose as bare codes)
_NUMCODE = re.compile(r"(?<![:_a-z0-9])\d{6,}\b")
# M3(i): axiom syntax has no business in prose annotations
_AXIOM_SYNTAX = re.compile(r"\b(SubClassOf|EquivalentTo|DisjointWith)\b|\b(some|only|exactly|min|max)\s+\d*\s*[A-Z{]")
# M3(ii): taxonomic claims must live in the axiom, not the annotation
_TAXONOMIC = re.compile(r"\bis a (kind|type|subclass) of\b|\bsubsumes\b|\bbroader than\b|\bparent class\b", re.I)
# M3(iii)/M4: curie-shaped tokens
_CURIE = re.compile(r"(?:cco|bfo|fhir|sysml|witsml|sdg):[A-Za-z0-9_]+")

TOKEN_FLOOR = 48          # the worklist floor — an authored surface must clear it
TOKEN_CEIL = 180          # don't blow up the registry's granularity (anchor p50 → window)


@dataclass
class Proposal:
    term: str
    alt_labels: list
    scope_note: str
    definition: str


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / (len(ta | tb) or 1)


def validate_annotation(template, prop: Proposal, *,
                        allowed_vocab: set, n_tokens: int,
                        self_retrieval: "tuple[str, float] | None" = None,
                        margin_floor: float = 0.05) -> "tuple[bool, str]":
    """M1–M6 over one proposal. ``n_tokens`` = the ColBERT token count of the surface
    AS ``topic_layer.term_anchor_text`` composes it with this proposal applied (the
    membrane judges exactly what MaxSim will see); ``allowed_vocab`` = lowercase tokens
    from the term's axiom vocabulary + glosses + name + grounding anchors;
    ``self_retrieval`` = (rank1_code, rel_margin_h) from MaxSim-ing the composed
    surface against the live registry (None skips M6-iii)."""
    text_all = " ".join([" ".join(prop.alt_labels), prop.scope_note, prop.definition])

    # M1 — shape
    if not (2 <= len(prop.alt_labels) <= 6):
        return False, f"M1: need 2-6 altLabels, got {len(prop.alt_labels)}"
    if any(len(a.split()) > 5 or not a.strip() for a in prop.alt_labels):
        return False, "M1: each altLabel must be 1-5 words, non-empty"
    if not prop.scope_note.strip() or not prop.definition.strip():
        return False, "M1: scope_note and definition must be non-empty"

    # M2 — token bounds vs the registry's granularity
    if n_tokens < TOKEN_FLOOR:
        return False, f"M2: composed surface is {n_tokens} tokens, floor is {TOKEN_FLOOR} — say more"
    if n_tokens > TOKEN_CEIL:
        return False, f"M2: composed surface is {n_tokens} tokens, ceiling is {TOKEN_CEIL} — tighten"

    # M3 — no smuggled ontological claims
    if _AXIOM_SYNTAX.search(text_all):
        return False, "M3: axiom syntax in prose — claims live in the Manchester axiom, not annotations"
    if _TAXONOMIC.search(text_all):
        return False, "M3: taxonomic phrasing ('is a kind of'/'subsumes'…) — assert subsumption in OWL or not at all"
    declared = set(_CURIE.findall(template.manchester_template or ""))
    for cu in _CURIE.findall(text_all):
        if cu not in declared:
            return False, f"M3: curie `{cu}` not declared by this term's axiom — no new references via prose"

    # M4 — vocabulary grounding (τ-style: most content words must be reachable)
    words = [w for w in re.findall(r"[a-z]{4,}", text_all.lower())]
    if words:
        hit = sum(1 for w in words if w in allowed_vocab)
        if hit / len(words) < 0.35:
            missing = sorted({w for w in words if w not in allowed_vocab})[:8]
            return False, (f"M4: only {hit}/{len(words)} content words ground in the term's "
                           f"declared vocabulary — ungrounded: {', '.join(missing)}")

    # M5 — clean room
    if _FORBIDDEN.search(text_all) or _NUMCODE.search(text_all):
        return False, "M5: clean-room tripwire (proprietary terminology or bare numeric code)"

    # M6 — discrimination
    pref = template.template_id.replace("_", " ").removeprefix("filler ")
    norm = {a.lower().replace("_", " ").strip() for a in prop.alt_labels}
    if pref.lower() in norm:
        return False, "M6: an altLabel merely repeats the prefLabel — add genuinely different surface forms"
    if _jaccard(prop.scope_note, template.verbal_template or "") > 0.8:
        return False, "M6: scope_note is a near-copy of the existing verbalization — add scope, not echo"
    if self_retrieval is not None:
        rank1, margin_h = self_retrieval
        if rank1 != template.template_id:
            return False, (f"M6: your surface retrieves `{rank1}` at rank-1, not this term — "
                           f"differentiate from that sibling explicitly")
        if margin_h < margin_floor:
            return False, (f"M6: self-retrieval margin_h {margin_h:.3f} < {margin_floor} — the surface "
                           f"is not yet distinguishable from its nearest non-ancestor competitor")
    return True, "ok"
