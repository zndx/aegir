#!/usr/bin/env python
"""demote_conflicting_domains — the arming gate's demotion pass (worklist 4).

kvasir witnesses on the ARMED union names, per refuted class, the axiom leaves of its
clash — including the `Domain:` lines of armed W1 frames. Every armed property cited in
any witness is DEMOTED (auto-band → review; withheld from the next arming round),
cumulatively. The loop:

    realize --arm-property-domains → kvasir witnesses --json → this script → repeat
    until kvasir finds no clash → HermiT signs the surviving armed set.

This is "mass verify as the gate" made executable at sub-second rounds: the deriver's
priced 7% auto-band error is pruned by MEASURED refutation, never by guesswork; demoted
properties stay CAS-review work (the escalation organ), not silent drops.

    components/kvasir/target/release/kvasir witnesses build/unified_armed/sdg-ontology.omn --json \\
      | uv run python scripts/demote_conflicting_domains.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build/domain_demotions.json"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_armed/sdg-ontology.omn")
    a = ap.parse_args()
    rep = json.load(sys.stdin)
    lines = a.omn.read_text().split("\n")

    def owner(line_no: int) -> "str | None":
        """kvasir cites the `Domain:`/`Range:` BODY line — the property name is the
        nearest preceding frame header. Resolve against the armed union text."""
        for i in range(min(line_no, len(lines)) - 1, -1, -1):
            hm = re.match(r"^(?:Object|Data)Property:\s*(sdg:[\w.-]+)", lines[i])
            if hm:
                return hm.group(1)
            if re.match(r"^(?:Class|Individual|Datatype|AnnotationProperty):", lines[i]):
                return None                    # crossed a non-property frame — not armed
        return None

    prior: "dict" = {"demoted": [], "rounds": []}
    if OUT.exists():
        prior = json.loads(OUT.read_text())
    demoted = set(prior.get("demoted", []))
    before = len(demoted)

    hits = 0
    for w in rep.get("witnesses", []):
        for ln, text in w.get("axioms", []):
            t = text.strip()
            m = re.match(r"^(?:Object|Data)Property:\s*(sdg:[\w.-]+)", t)
            if m:
                demoted.add(m.group(1))
                hits += 1
                continue
            if re.match(r"^(?:Domain|Range):\s*(?:bfo|cco|sdg):[\w.-]+$", t):
                p_ = owner(ln)
                if p_:
                    demoted.add(p_)
                    hits += 1
    new = len(demoted) - before
    prior["demoted"] = sorted(demoted)
    prior.setdefault("rounds", []).append(
        {"witnesses": rep.get("n_witnesses", 0),
         "unsat_classes": rep.get("n_unsat_classes", 0),
         "newly_demoted": new, "leaf_hits": hits})
    OUT.write_text(json.dumps(prior, indent=1))
    print(f"demoted {new} new properties (total {len(demoted)}) from "
          f"{rep.get('n_unsat_classes', 0)} unsat · → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
