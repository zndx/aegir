"""relation_signatures — BFO-faithful signatures for the numeric backbone + grounding for sdg coins (#27).

The realizer's NUMERIC_BFO declares BFO object properties BARE (no domain/range), and sdg relation
coins are declared bare too — so a role that "realizes" an organization gives HermiT nothing to refuse
(the verbalization audit's finding). This module supplies the missing halves, per doctrine (never mint
in bfo:/cco: — ground sdg: coins via rdfs:subPropertyOf to the REAL BFO IRIs):

  * ``SIGNATURES_OMN`` — Domain/Range/InverseOf frames for the BFO-2020 numeric properties the
    realizer already declares (realizes/realized-in, participates/has-participant, bearer-of/
    inheres-in, parthood pairs), stated against the numeric class backbone.
  * ``GROUNDING`` — sdg relation-name stems → the BFO property they borrow their name from; the
    coin becomes ``SubPropertyOf`` its nominal parent and INHERITS the signature.
  * ``grounding_frames(props)`` — the Manchester frames for a set of used sdg property locals.

ARMING IS STAGED (``AEGIR_RELATION_SIGNATURES=1``): switching this on makes the ~39 lint-confirmed
miscasts UNSAT AT REALIZE — the intended signal (Sweep-B doctrine: unsat are signals, not drops) —
so it stays OFF for production realize until move 2 re-authors those axioms. scripts/
verify_relation_signatures.py proves the arming fires without touching production.

Long-tail singleton coins (certifiesAcademicStanding, monitorsEligibilityStatus, …) are NOT
auto-grounded — each needs a semantic decision (move 2's CAS loop); they stay bare and listed.
"""
from __future__ import annotations

import re

# ── BFO-2020 signatures over the numeric backbone (classes the realizer declares) ────────────
# 0000015 process · 0000017 realizable · 0000002 continuant · 0000003 occurrent ·
# 0000004 independent continuant · 0000020 specifically dependent continuant
SIGNATURES_OMN = """
ObjectProperty: bfo:0000055
    Domain: bfo:0000015
    Range: bfo:0000017
    InverseOf: bfo:0000054
ObjectProperty: bfo:0000054
    Domain: bfo:0000017
    Range: bfo:0000015
ObjectProperty: bfo:0000056
    Domain: bfo:0000002
    Range: bfo:0000015
    InverseOf: bfo:0000057
ObjectProperty: bfo:0000057
    Domain: bfo:0000015
    Range: bfo:0000002
ObjectProperty: bfo:0000196
    Domain: bfo:0000004
    Range: bfo:0000020
    InverseOf: bfo:0000197
ObjectProperty: bfo:0000197
    Domain: bfo:0000020
    Range: bfo:0000004
ObjectProperty: bfo:0000176
    Domain: bfo:0000002
    Range: bfo:0000002
    InverseOf: bfo:0000178
ObjectProperty: bfo:0000178
    Domain: bfo:0000002
    Range: bfo:0000002
ObjectProperty: bfo:0000132
    Domain: bfo:0000003
    Range: bfo:0000003
    InverseOf: bfo:0000117
ObjectProperty: bfo:0000117
    Domain: bfo:0000003
    Range: bfo:0000003
ObjectProperty: bfo:0000066
    Domain: bfo:0000003
    Range: bfo:0000004
"""

# sdg coin stem → the BFO property whose name it borrows (SubPropertyOf → signature inherited).
# Order matters: more specific stems first ('realizedIn' before 'realiz').
_GROUNDING_STEMS: "list[tuple[re.Pattern, str]]" = [
    (re.compile(r"^realized[_]?in$", re.I), "bfo:0000054"),
    (re.compile(r"^realizes[_]?in$", re.I), "bfo:0000054"),          # 'realizesIn' reads as realized-in
    (re.compile(r"^(role[_]?realizes|realizes.*)$", re.I), "bfo:0000055"),
    # membership is NOT bfo participation (RH adjudication 2026-07-22: a group is a
    # continuant — the relation is member-parthood): before the participates catch-all.
    (re.compile(r"^participates[_]?in[_]?group$", re.I), "bfo:0000129"),
    (re.compile(r"^(borne[_]?by|has[_]?bearer.*|inheres.*)$", re.I), "bfo:0000197"),
    (re.compile(r"^bears.*$", re.I), "bfo:0000196"),
    (re.compile(r"^participates[_]?in.*$", re.I), "bfo:0000056"),
    (re.compile(r"^has[_]?participant.*$", re.I), "bfo:0000057"),
    (re.compile(r"^(has[_]?part|contains.*part.*)$", re.I), "bfo:0000178"),
    (re.compile(r"^part[_]?of$", re.I), "bfo:0000176"),
    (re.compile(r"^occurs[_]?in$", re.I), "bfo:0000066"),
]


# Expected (subject, filler) CATEGORY sides per grounded BFO property — the signature
# the usage-directed grounding checks each site against (RH adjudication 2026-07-22:
# conforming sites ground; mirror-conforming partOf sites split to occurrent part-of;
# contradicting sites are ESCALATED, never silently typed). Sides: cont | occ | any.
BFO_SIGNATURES: "dict[str, tuple[str, str]]" = {
    "bfo:0000054": ("cont", "occ"),    # realized in: realizable bearer → process
    "bfo:0000055": ("occ", "cont"),    # realizes: process → realizable
    "bfo:0000196": ("cont", "cont"),   # bears
    "bfo:0000197": ("cont", "cont"),   # inheres in / borne by
    "bfo:0000056": ("cont", "occ"),    # participates in
    "bfo:0000057": ("occ", "cont"),    # has participant
    "bfo:0000178": ("cont", "cont"),   # has continuant part
    "bfo:0000176": ("cont", "cont"),   # continuant part of
    "bfo:0000129": ("cont", "cont"),   # member part of (membership)
    "bfo:0000132": ("occ", "occ"),     # occurrent part of
    "bfo:0000066": ("occ", "any"),     # occurs in (site/spatial region fillers vary)
}

# partOf's mereology polysemy: occ→occ usage sites belong to occurrent part-of —
# the usage-directed SPLIT coins the sibling (HermiT-grounded, per the coined-alias rule).
MIRROR_SPLIT: "dict[str, tuple[str, str]]" = {
    # grounded bfo → (mirror bfo, sdg coin for mirror-conforming sites)
    "bfo:0000176": ("bfo:0000132", "occurrentPartOf"),
}


def grounding_of(local: str) -> "str | None":
    for pat, bfo in _GROUNDING_STEMS:
        if pat.match(local):
            return bfo
    return None


def grounding_frames(props: "set[str] | list[str]") -> "tuple[str, list[str]]":
    """Manchester frames grounding each sdg property local to its nominal BFO parent.
    → (omn_text, ungrounded_locals). Ungrounded = the long tail needing move-2 decisions."""
    lines, loose = [], []
    for p in sorted(set(props)):
        bfo = grounding_of(p)
        if bfo:
            lines.append(f"ObjectProperty: sdg:{p}\n    SubPropertyOf: {bfo}")
        else:
            loose.append(p)
    return "\n".join(lines), loose
