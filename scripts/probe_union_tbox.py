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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_realize/sdg-ontology.omn")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=REPO / "build/union_tbox_probe.json")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("et", REPO / "scripts/emit_taxonomy.py")
    assert spec and spec.loader
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    omn = a.omn.read_text()
    tbox = strip_abox(omn)
    n_ind = len(re.findall(r"^Individual:", omn, flags=re.M))
    print(f"TBox probe: {len(tbox)} chars (ABox stripped: {n_ind} individuals)", flush=True)
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
