#!/usr/bin/env python
"""strengthen_differentiae — a TARGETED re-authoring pass over the weak (globally-common-value) differentiae.

Discriminability flagged that ~48 species lean on generic status values ('active'×25, 'pending', 'completed'…)
that distinguish within a genus but characterise nothing across the corpus. This re-fires the CAS loop on
exactly those, with (a) a nudge toward a MORE SPECIFIC differentia — ideally an object property to a
characteristic related class (the multi-table discriminators the DEE eval prizes) — and (b) a DISTINCTIVENESS
membrane that rejects any value still in the globally-common set. Updates the genera JSON in place.

    env $(cat build/cuda-driver-libs/.env|xargs) uv run --no-sync python scripts/strengthen_differentiae.py \
        --taxonomy build/taxonomy/v05 --entities <run>/entities --min-shared 4
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.differentia_authoring import (engine_proposer, observability_membrane,  # noqa: E402
                                                  parse_membrane)
from aegir.ontology.genus_induction import load_specs  # noqa: E402


def common_values(genera: "list[dict]", min_shared: int) -> "set[str]":
    c = Counter()
    for r in genera:
        for d in r["differentiae"].values():
            if d["accepted"]:
                vm = re.search(r'value\s+"([^"]+)"', d["restriction"])
                if vm:
                    c[vm.group(1)] += 1
    return {v for v, n in c.items() if n >= min_shared}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default=str(REPO / "build" / "taxonomy" / "v05"))
    ap.add_argument("--entities", required=True)
    ap.add_argument("--min-shared", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    specs = load_specs(Path(a.entities))
    class_names = frozenset(specs)
    files = sorted(glob.glob(str(Path(a.taxonomy) / "genera" / "*.json")))
    genera = [json.loads(Path(f).read_text()) for f in files]
    common = common_values(genera, a.min_shared)
    print(f"strengthening: {len(common)} generic values → targeting weak differentiae", flush=True)

    strengthened = 0
    for f, r in zip(files, genera):
        g = r["genus"]
        members = [(s, specs.get(s, s)) for s in r["differentiae"]]
        changed = False
        for s, d in r["differentiae"].items():
            if not d["accepted"]:
                continue
            vm = re.search(r'value\s+"([^"]+)"', d["restriction"])
            if not (vm and vm.group(1) in common):
                continue                                   # only the weak ones
            sibs = [(n, dn) for n, dn in members if n != s]
            feedback = (f'your differentia "{d["property"]} value \\"{vm.group(1)}\\"" is GENERIC — that value is '
                        f"shared by many species and characterises nothing. Author a MORE SPECIFIC differentia "
                        f"that captures what ESSENTIALLY makes {s} distinct: PREFER an object property (some "
                        f"CharacteristicClass it relates to), or a rare characteristic value — not a status.")
            for _ in range(a.rounds):
                try:
                    cand = engine_proposer(g, s, specs.get(s, ""), sibs, feedback, None)
                except Exception:  # noqa: BLE001
                    break
                ok, why = parse_membrane(cand)
                if ok:
                    ok, why = observability_membrane(cand, class_names)
                if ok:
                    vm2 = re.search(r'value\s+"([^"]+)"', cand.get("restriction", ""))
                    if vm2 and vm2.group(1) in common:       # distinctiveness membrane
                        ok, why = False, f'"{vm2.group(1)}" is still a generic shared value — be more specific'
                if ok:
                    d.update({"property": cand["property"], "kind": cand["kind"],
                              "restriction": cand["restriction"], "why": cand.get("why", ""),
                              "reason": "strengthened (distinctiveness)", "rounds": d.get("rounds", 1) + 1})
                    strengthened += 1
                    changed = True
                    break
                feedback = why
        if changed:
            Path(f).write_text(json.dumps(r, indent=1))
            print(f"  {g}: strengthened", flush=True)
    print(f"\nstrengthened {strengthened} weak differentiae → {a.taxonomy}/genera", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
