#!/usr/bin/env python
"""escalate_grounding_sites — the grounding-grain measured-refutation pass.

For every kvasir-witnessed unsat class, find its usage sites of GROUNDED (or split)
properties and append them to build/grounding_site_escalations.json — the ledger the
signature-checked grounding pass consumes on the next realize (those sites rewrite to
<prop>__escalated, declared but ungrounded). The checker in the loop is kvasir at theory
strength, not the local category lift (the ConferenceMarketplace lesson: a "?"-side
subject can be theory-determined).

    components/kvasir/target/release/kvasir witnesses <doc> --json \\
      | uv run python scripts/escalate_grounding_sites.py --omn <doc>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "build/grounding_site_escalations.json"

_IRI2PFX = [
    (re.compile(r"<https://signals\.zndx\.org/sdg#([\w.-]+)>"), r"sdg:\1"),
    (re.compile(r"<http://purl\.obolibrary\.org/obo/BFO_(\d{7})>"), r"bfo:\1"),
    (re.compile(r"<https://www\.commoncoreontologies\.org/(ont\d+)>"), r"cco:\1"),
]


def norm(t: str) -> str:
    for rx, rep in _IRI2PFX:
        t = rx.sub(rep, t)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_grounded/sdg-ontology.omn")
    a = ap.parse_args()

    from aegir.ontology.relation_signatures import grounding_of, MIRROR_SPLIT
    coins = {coin for _, coin in MIRROR_SPLIT.values()}

    rep = json.load(sys.stdin)
    unsat = {w["name"] for w in rep.get("witnesses", []) if w["kind"] == "unsat_class"}
    omn = a.omn.read_text()

    prior = json.loads(OUT.read_text()) if OUT.exists() else {"sites": []}
    seen = {(s_["class"], s_["property"], s_["filler"]) for s_ in prior["sites"]}
    frame_rx = re.compile(r"^Class:\s*(<[^>\n]+>|(?:sdg|bfo|cco):[\w.-]+)([^\n]*)\n"
                          r"((?:    [^\n]*\n)*)", re.M)
    n_new = 0
    for m in frame_rx.finditer(omn):
        cls = norm(m.group(1))
        if cls.split(":", 1)[-1] not in unsat and cls not in unsat:
            continue
        body = m.group(2) + "\n" + m.group(3)
        for um in re.finditer(r"sdg:(\w+)\s+(?:some|only|exactly \d+|min \d+|max \d+)\s+(\S+)",
                              body):
            q, filler = um.group(1), norm(um.group(2)).rstrip(",")
            base = q[:-len("__escalated")] if q.endswith("__escalated") else q
            if not (grounding_of(base) or base in coins):
                continue
            # ledger keys use the ORIGINAL property name (the pass checks pre-rewrite)
            key = (cls, base, filler)
            if key in seen:
                continue
            seen.add(key)
            prior["sites"].append({"class": cls, "property": base, "filler": filler,
                                   "witnessed": True})
            n_new += 1
    OUT.write_text(json.dumps(prior, indent=1))
    print(f"escalated {n_new} new grounding sites (ledger total {len(prior['sites'])}) "
          f"from {len(unsat)} unsat classes → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
