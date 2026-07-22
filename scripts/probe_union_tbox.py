#!/usr/bin/env python
"""probe_union_tbox — TBox-only HermiT verdict on the union OMN (shakedown instrument).

Strips Individual: frames (the ABox clash pass is its own worklist item) and counts
unsatisfiable classes. Phase-1 baseline: 1,836 unsat from 5 double-category roots.
Phase-2 (post --reconcile-categories) expects the cascade to collapse.

    uv run python scripts/probe_union_tbox.py [--omn build/unified_realize/sdg-ontology.omn]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def strip_abox(omn: str) -> str:
    """Remove Individual frames AND the ObjectOneOf enumeration body lines that reference
    them (undeclared names in { } fail the OMN parse — the phase-1 lesson). Annotation
    strings containing literal { } braces are untouched: the pattern requires the whole
    body line to be a pure enum."""
    omn = re.sub(r"^Individual:[^\n]*\n(?:    [^\n]*\n|\n(?=    ))*", "", omn, flags=re.M)
    return re.sub(r"^\s*(?:EquivalentTo|SubClassOf):\s*\{[^}]*\}\s*\n", "", omn, flags=re.M)


def _split_items(rest: str) -> "list[str]":
    items, depth, instr, cur = [], 0, False, ""
    for i, ch in enumerate(rest):
        if ch == '"' and (i == 0 or rest[i - 1] != "\\"):
            instr = not instr
        elif not instr:
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth -= 1
            elif ch == "," and depth == 0:
                items.append(cur)
                cur = ""
                continue
        cur += ch
    if cur.strip():
        items.append(cur)
    return items


def taxonomy_core(omn: str) -> str:
    """Reduce to the bare taxonomy: only bare atoms (and bare conjuncts of ≡-intersections,
    lifted to SubClassOf) survive in SubClassOf/EquivalentTo sections. Restrictions gone →
    ∃/min-card propagation gone → the residual unsat set IS the contradiction core."""
    bare = re.compile(r"^(?:bfo:\d{7}|cco:\w+|sdg:[\w.-]+|<[^>]+>)$")
    sect = re.compile(r"^(\s*)(SubClassOf|EquivalentTo):\s*(.*)$")
    # single-line frames (Class: <iri> EquivalentTo: ...) carry their section mid-line —
    # normalize to two-line form so the reduction below sees them.
    omn = re.sub(r"^(Class:\s*(?:<[^>\n]+>|[\w:.-]+))\s+((?:SubClassOf|EquivalentTo):[^\n]*)$",
                 r"\1\n    \2", omn, flags=re.M)
    out = []
    for line in omn.split("\n"):
        m = sect.match(line)
        if not m:
            out.append(line)
            continue
        keep = []
        for it in _split_items(m.group(3)):
            s = it.strip()
            if bare.match(s):
                keep.append(s)
            elif m.group(2) == "EquivalentTo" and " and " in s:
                for c in re.split(r"\s+and\s+", s):
                    c = c.strip().strip("()")
                    if bare.match(c):
                        keep.append(c)
        if keep:
            out.append(f"{m.group(1)}SubClassOf: " + ", ".join(sorted(set(keep))))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_realize/sdg-ontology.omn")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=REPO / "build/union_tbox_probe.json")
    ap.add_argument("--taxonomy-core", action="store_true",
                    help="probe the bare taxonomy only (no restrictions → no propagation; "
                         "the unsat set is the contradiction core)")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("et", REPO / "scripts/emit_taxonomy.py")
    assert spec and spec.loader
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    omn = a.omn.read_text()
    tbox = strip_abox(omn)
    if a.taxonomy_core:
        tbox = taxonomy_core(tbox)
    n_ind = len(re.findall(r"^Individual:", omn, flags=re.M))
    print(f"TBox probe: {len(tbox)} chars (ABox stripped: {n_ind} individuals"
          f"{'; taxonomy core' if a.taxonomy_core else ''})", flush=True)
    r = et.hermit(tbox, budget_s=a.budget)
    r["individuals_stripped"] = n_ind
    a.out.write_text(json.dumps(r, indent=1))
    unsat = r.get("unsat") or []
    print(f"consistent={r.get('consistent')} n_classes={r.get('n_classes')} "
          f"unsat={len(unsat)}")
    if unsat:
        print("sample:", [u.rsplit("#", 1)[-1] for u in unsat[:8]])
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
