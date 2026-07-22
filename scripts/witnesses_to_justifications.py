#!/usr/bin/env python
"""witnesses_to_justifications — kvasir witnesses → the justification shape the cutter eats.

The repair loop's explain step becomes: `kvasir witnesses <union> --json` (whole-corpus,
sub-second) → this adapter → build/taxcore_justifications.json → derive_category_cuts.py
unchanged. HermiT stays the certificate (the loop's probe step); kvasir carries the
justification load BlackBox/HST could not (20-min per-class timeouts, ABox refusal).

Witness axioms arrive as (source_line, text) pairs of the ORIGINAL Manchester lines; the
cutter's rules parse functional-syntax-shaped strings, so each cited line is re-rendered
into the SubClassOf(...)/EquivalentClasses(...) forms the rules match on.

    components/kvasir/target/release/kvasir witnesses build/unified_realize/sdg-ontology.omn --json \\
      | uv run python scripts/witnesses_to_justifications.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build/taxcore_justifications.json"
SDG = "https://signals.zndx.org/sdg#"

_IRI2LOCAL = [
    (re.compile(r"<https://signals\.zndx\.org/sdg#([\w.-]+)>"), r"\1"),
    (re.compile(r"<http://purl\.obolibrary\.org/obo/BFO_(\d{7})>"), r"BFO_\1"),
    (re.compile(r"<https://www\.commoncoreontologies\.org/(ont\d+)>"), r"\1"),
    (re.compile(r"\bsdg:([\w.-]+)"), r"\1"),
    (re.compile(r"\bbfo:(\d{7})"), r"BFO_\1"),
    (re.compile(r"\bcco:(ont\d+)"), r"\1"),
]


def local(t: str) -> str:
    for rx, rep in _IRI2LOCAL:
        t = rx.sub(rep, t)
    return t


def render(line: str, subject_hint: "str | None") -> "list[str]":
    """One cited Manchester line → functional-syntax-shaped strings for the cutter."""
    capped = len(line) >= 239         # kvasir cites at most 240 chars — the tail is mid-token
    line = local(line.strip())
    out = []
    m = re.match(r"^DisjointClasses:\s*(\S+?),\s*(\S+)$", line)
    if m:
        return [f"DisjointClasses({m.group(1)} {m.group(2)})"]
    # 'Class: X SubClassOf: ...' single-line or 'SubClassOf: ...' body line (subject from hint)
    cm = re.match(r"^Class:\s*(\S+?)(?:\s+(.*))?$", line)
    subject = subject_hint
    rest = line
    if cm:
        subject = cm.group(1)
        rest = cm.group(2) or ""
    sm = re.match(r"^(SubClassOf|EquivalentTo):\s*(.*)$", rest)
    if not (sm and subject):
        return out
    body = sm.group(2)
    if sm.group(1) == "EquivalentTo" and " and " in body:
        genus = body.split(" and ", 1)[0].strip().strip("()")
        if re.fullmatch(r"[\w.-]+", genus):
            out.append(f"EquivalentClasses({subject} ObjectIntersectionOf({genus} …))")
    # bare top-level atoms → SubClassOf pairs
    depth = 0
    cur, items = "", []
    for ch in body:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(cur)
            cur = ""
            continue
        cur += ch
    items.append(cur)
    if capped:
        items = items[:-1]        # citation hit kvasir's 240-char cap — drop the cut tail
    for it in items:
        it = it.strip()
        if re.fullmatch(r"[\w.-]+", it):
            out.append(f"SubClassOf({subject} {it})")
    return out


def main() -> int:
    rep = json.load(sys.stdin)
    why = {}
    for w in rep.get("witnesses", []):
        if w["kind"] != "unsat_class":
            continue                      # class repairs first; RInst heals with them
        axioms: "list[str]" = []
        for _ln, text in w["axioms"]:
            # indented body lines carry no Class: prefix — the witness's own name is the
            # best subject hint (correct for the final-step subject; foreign body lines
            # in the leaf set still render their Class:-prefixed forms exactly)
            axioms.extend(render(text, w["name"]))
        if axioms:
            why[f"{SDG}{w['name']}"] = {"axioms": sorted(set(axioms)),
                                        "why": f"kvasir {w['family']}"}
    OUT.write_text(json.dumps(
        {"unsat_n": rep.get("n_unsat_classes", len(why)), "why": why,
         "source": "kvasir witnesses (sound-refutation sweep)"}, indent=1))
    print(f"{len(why)} justifications ← {rep.get('n_witnesses', 0)} witnesses "
          f"({rep.get('saturate_ms', '?')} ms saturation) → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
