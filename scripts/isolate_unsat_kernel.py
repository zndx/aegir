#!/usr/bin/env python
"""isolate_unsat_kernel — split the unsat set into KERNEL (local deaths) vs victims.

With the taxonomy core dry, a class is unsatisfiable either LOCALLY (restriction
interactions on its own frame — functional-property intersections, domain/range forcing)
or by REFERENCE (∃/card restrictions or bare parents reaching another unsat class).
The kernel members are the repair targets; their HermiT justifications are local and
cheap where full-web explanations explode. Pure text — no reasoner.

    uv run python scripts/isolate_unsat_kernel.py
    → build/unsat_kernel.json (kernel + per-victim witness chains sample)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

UNION = REPO / "build/unified_realize/sdg-ontology.omn"
PROBE = REPO / "build/union_tbox_probe.json"
OUT = REPO / "build/unsat_kernel.json"

_IRI2PFX = [
    (re.compile(r"<https://signals\.zndx\.org/sdg#([\w.-]+)>"), r"sdg:\1"),
    (re.compile(r"<http://purl\.obolibrary\.org/obo/BFO_(\d{7})>"), r"bfo:\1"),
    (re.compile(r"<https://www\.commoncoreontologies\.org/(ont\d+)>"), r"cco:\1"),
]
_FRAME = re.compile(r"^Class:\s*(<[^>\n]+>|(?:sdg|bfo|cco):[\w.-]+)([^\n]*)\n"
                    r"((?:    [^\n]*\n)*)", re.M)
_SECT = re.compile(r"^\s*(?:SubClassOf|EquivalentTo):\s*(.*)$")
_ATOM = r"(?:bfo:\d{7}|cco:\w+|sdg:[\w.-]+)"
# a named class mentioned as restriction filler or bare item
_REF = re.compile(rf"(?:^|[\s,(])({_ATOM})(?=$|[\s,)])")


def norm(t: str) -> str:
    for rx, rep in _IRI2PFX:
        t = rx.sub(rep, t)
    return t


def main() -> int:
    omn = UNION.read_text()
    unsat = {("sdg:" + u.rsplit("#", 1)[-1]) for u in json.loads(PROBE.read_text())["unsat"]}
    refs: "dict[str, set]" = {}
    for m in _FRAME.finditer(omn):
        cls = norm(m.group(1))
        rs = refs.setdefault(cls, set())
        for line in ([m.group(2)] if m.group(2).strip() else []) + m.group(3).splitlines():
            sm = _SECT.match(line)
            if not sm:
                continue
            body = norm(sm.group(1))
            body = re.sub(r'"[^"]*"', '""', body)          # strings out
            for t in _REF.findall(body):
                if t != cls and not re.match(r"^(?:some|only|min|max|exactly|value|and|or|not|that|Self)$", t):
                    rs.add(t)
    kernel = sorted(u for u in unsat if not (refs.get(u, set()) & unsat))
    # victims sample: which unsat classes they reference (the chain witnesses)
    victims = {u: sorted(refs.get(u, set()) & unsat)[:4]
               for u in sorted(unsat - set(kernel))[:12]}
    OUT.write_text(json.dumps({"n_unsat": len(unsat), "kernel": kernel,
                               "victims_sample": victims}, indent=1))
    print(f"unsat {len(unsat)} → kernel {len(kernel)} (local deaths; the repair targets)")
    for k in kernel[:15]:
        print("  ", k)
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
