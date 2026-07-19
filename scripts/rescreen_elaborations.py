#!/usr/bin/env python
"""rescreen_elaborations — retrofit the strengthened membrane onto the accepted elaboration stock.

The v2 membrane's CLT floor is DIRECTION-BLIND (feature-overlap Jaccard sees the same entities and
relations whether 'X is requested by Y' or 'X requests Y'), and candidates clearing the floor never
reached the judge. The scorer's head-to-head losses exposed accepted direction-flips. The membrane
now routes subject-drifted candidates to a direction-aware judge; this script applies the SAME
screen retroactively to every pool member:

  * dropped-shape purge — mechanical members of census-flagged dragging shapes
    (relates-via / relation-connects / denotes / qualifies) leave the pools outright;
  * drift re-screen — members not opening with the base's head slot are judged (direction-aware
    prompt); judge-unfaithful members are PRUNED FROM THE POOL and their template lands on the
    ACP worklist with the judge's reason (inform-and-refine: never a silent drop).

    uv run python scripts/rescreen_elaborations.py            # engine must be serving
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_SLOT_RE = re.compile(r"\{[^}]*\}")
_OPENERS = re.compile(r"^(?:a|an|the|each|every|any|all)\s+", re.I)
_DROP = [re.compile(p) for p in (r"^Any \{?\w+\}? necessarily relates\b",
                                 r"^The .+ relation connects\b",
                                 r"^\{?\w+\}? denotes\b",
                                 r"^Any \{?\w+\}? qualifies as\b")]
_WORKLIST = REPO / "build/elaboration_acp_worklist.json"
FAITHFUL_FLOOR = 4


def main() -> int:
    from aegir.ontology.verbalization_quality import judge
    cat_path = REPO / "src/aegir/ontology/catalog/catalog.json"
    cat = json.loads(cat_path.read_text())
    tpls = cat.get("templates", cat if isinstance(cat, list) else [])
    worklist = json.loads(_WORKLIST.read_text()) if _WORKLIST.exists() else {}

    n_drop = n_judged = n_pruned = n_kept = 0
    for t in tpls:
        fr = t.get("verbal_templates") or []
        if len(fr) < 2:
            continue
        base, rest = fr[0], fr[1:]
        heads = _SLOT_RE.findall(base)
        head = heads[0] if heads else None
        keep = [base]
        pruned_here = []
        for f in rest:
            if any(p.search(f) for p in _DROP):
                n_drop += 1
                continue
            if head and not _OPENERS.sub("", f.strip()).startswith(head):
                v = judge(t.get("manchester_template") or base, f, base, cand_is_a=True)
                n_judged += 1
                if v is None:
                    keep.append(f)               # engine hiccup: hold, do not prune blind
                    continue
                if (v.get("faithful") or 0) < FAITHFUL_FLOOR:
                    n_pruned += 1
                    pruned_here.append({"frame": f, "faithful": v.get("faithful"),
                                        "why": str(v.get("why"))[:160]})
                    continue
            keep.append(f)
        if pruned_here:
            key = f"{t['template_id']}::rescreen::direction-2026-07-19"
            worklist[key] = {"template_id": t["template_id"], "base": base,
                             "reasons": "direction/subject-drift pruned by re-screen: "
                                        + "; ".join(x["why"] or "unfaithful" for x in pruned_here)[:400],
                             "pruned": pruned_here}
        if len(keep) != len(fr):
            t["verbal_templates"] = keep
            t["verbal_template"] = keep[0]
        n_kept += len(keep) - 1
        if (n_judged and n_judged % 25 == 0):
            print(f"  judged {n_judged} · pruned {n_pruned}", flush=True)

    cat_path.write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    _WORKLIST.write_text(json.dumps(worklist, indent=1))
    print(f"shape-purged {n_drop} · drift-judged {n_judged} · pruned {n_pruned} · "
          f"kept {n_kept} non-base members → catalog + worklist ({len(worklist)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
