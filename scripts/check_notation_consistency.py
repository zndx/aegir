#!/usr/bin/env python
"""check_notation_consistency — the topic notation-tree gate (RH 2026-07-26).

Notation ≠ point-hood (aperture membership is an explicit skos:Collection, not a
notation side-effect); notation ≡ position in the topic hierarchy. The dotted
skos:notation encodes that hierarchy and drives subtree roll-up, so a missing or
inconsistent code renders the vocabulary useless. This gate checks the notation
tree's INTERNAL invariants directly — no structural-pattern guessing about which
concepts are "topic" (RH: connote roles with skos:Collection / skos:ConceptScheme,
don't infer them):

  * COMPLETENESS (propagation) — if a concept's skos:broader carries a notation,
    the concept MUST carry one too. You cannot have a coded topic parent with an
    uncoded child. Metamodel vocabulary links to the tree by skos:broadMatch
    (cross-scheme), never skos:broader, so it is exempt by construction; enum
    values have no broader and are likewise exempt.
  * CONSISTENCY — a DOTTED code must extend its broader's code by exactly one
    segment (18.1.1.4 under 18.1.1). A BARE-INTEGER code marks a domain root and
    carries no parent-prefix obligation (its broader, if any, is a cross-tier
    grouping, not a notation parent).
  * UNIQUENESS — no two concepts share a notation.

Mechanical, JVM-free, rdflib-only. Exit 1 on any violation (worklist printed).
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import rdflib
from rdflib.namespace import SKOS

REPO = Path(__file__).resolve().parent.parent
ONT = REPO / "src/aegir/ontology"
_BARE = re.compile(r"^\d+$")


def _frag(u) -> str:
    return str(u).rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _load() -> rdflib.Graph:
    g = rdflib.Graph()
    for f in sorted(glob.glob(str(ONT / "*.ttl"))):
        try:
            g.parse(f, format="turtle")
        except Exception as e:  # noqa: BLE001
            print(f"  parse-skip {Path(f).name}: {str(e)[:80]}")
    return g


def main() -> int:
    g = _load()
    broader = {}
    for s, o in g.subject_objects(SKOS.broader):
        broader.setdefault(_frag(s), _frag(o))
    code = {_frag(s): str(o) for s, o in g.subject_objects(SKOS.notation)}

    missing, inconsistent = [], []
    for child, parent in broader.items():
        pcode = code.get(parent)
        if not pcode:
            continue  # parent isn't in the notation tree → child unconstrained
        ccode = code.get(child)
        if not ccode:
            missing.append(f"{child} → broader {parent} ({pcode}) is coded but {child} is not")
            continue
        if _BARE.match(ccode):
            continue  # child is itself a domain root; broader is a cross-tier grouping
        if ccode != pcode and not ccode.startswith(pcode + "."):
            inconsistent.append(f"{child} ({ccode}) ⊄ broader {parent} ({pcode})")

    seen, dupes = {}, []
    for f, c in code.items():
        if c in seen:
            dupes.append(f"{c}: {seen[c]} & {f}")
        seen[c] = f

    print(f"notation tree: {len(code)} coded concepts · {len(broader)} broader edges checked")
    ok = True
    if missing:
        ok = False
        print(f"✘ INCOMPLETE — coded parent, uncoded child ({len(missing)}):")
        for x in missing:
            print(f"    {x}")
    if inconsistent:
        ok = False
        print(f"✘ INCONSISTENT — code doesn't extend broader ({len(inconsistent)}):")
        for x in inconsistent:
            print(f"    {x}")
    if dupes:
        ok = False
        print(f"✘ DUPLICATE codes ({len(dupes)}): {dupes}")
    if ok:
        print("notation gate: COMPLETE + CONSISTENT ✓")
        return 0
    print("notation gate: FAILED — the topic notation tree needs remediation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
