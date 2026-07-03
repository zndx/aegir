#!/usr/bin/env python
"""Registry remediation purge — the deterministic half of the re-seed loop.

Two flagged classes leave ``individual_registry.json`` (the re-seed under the hardened membranes
regenerates their columns):

  1. REAL-WORLD PARTICULARS (``individuals.real_entity_hits`` — the fictional-particulars policy):
     the corpus asserts nothing about real organizations/products/people.
  2. CROSS-TEMPLATE LABEL COLLISIONS (``build/abox_clash_signals.json``, written by the realize):
     the same label admitted under classes that clash under BFO+π(CCO) becomes ONE multi-typed
     individual and is withheld at realize. Purged from EVERY template that carries it so the
     re-seed (with the admit-time collision membrane now live) coins distinct names per class.

Deterministic, idempotent, loud: prints per-template drops, writes ``build/reseed_targets.json``
(the ``seed_individuals --only`` input), and leaves untouched values alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import individuals as IND  # noqa: E402
from aegir.ontology.individuals import _slug, real_entity_hits  # noqa: E402

CLASH_SIGNALS = REPO / "build/abox_clash_signals.json"
TARGETS_OUT = REPO / "build/reseed_targets.json"


def main() -> int:
    reg = IND.load_registry()
    clash_slugs: set[str] = set()
    if CLASH_SIGNALS.exists():
        for entry in json.loads(CLASH_SIGNALS.read_text()):
            clash_slugs.update(entry.get("individuals", []))

    n_real = n_clash = 0
    targets: dict[str, list[str]] = {}
    for tid, rec in reg.get("templates", {}).items():
        for col, vals in list((rec.get("columns") or {}).items()):
            flagged = {v for v, _b in real_entity_hits(list(vals))}
            n_real += len(flagged)
            coll = {v for v in vals if f"i_{_slug(v)}" in clash_slugs}
            n_clash += len(coll - flagged)
            drop = flagged | coll
            if not drop:
                continue
            rec["columns"][col] = [v for v in vals if v not in drop]
            targets.setdefault(tid, []).append(col)
            why = []
            if flagged:
                why.append(f"{len(flagged)} real-particular")
            if coll - flagged:
                why.append(f"{len(coll - flagged)} collision")
            print(f"  {tid}.{col}: dropped {len(drop)} ({', '.join(why)}) → {len(rec['columns'][col])} kept")

    IND.save_registry(reg)
    TARGETS_OUT.write_text(json.dumps(sorted(targets), indent=1) + "\n")
    print(f"\npurged: {n_real} real-particular value(s) + {n_clash} collision value(s) "
          f"across {len(targets)} template(s)")
    print(f"re-seed targets → {TARGETS_OUT.relative_to(REPO)}")
    print(f"next: uv run --no-sync python scripts/seed_individuals.py --only "
          f"$(python3 -c \"import json; print(','.join(json.load(open('{TARGETS_OUT}'))))\") --rounds 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
