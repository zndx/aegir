#!/usr/bin/env python
"""lint_relation_signatures — deterministic subject-category × relation × object-category lint over the
catalog templates, against ACCEPTED BFO/RO relation signatures.

The verbalization audit (RH 2026-07-19) traced fluent-but-miscast prose to the template-era axioms:
``sdg:``-coined mirrors of BFO relation names (sdg:realizes …) carry NONE of BFO's domain/range, so
HermiT has nothing to refute when an independent continuant "realizes" a sequence or a role "realizes"
an organization. This lint reconstructs the verdicts BFO's signatures would give:

  realizes        process → realizable entity (role/disposition/function)     [BFO_0000055]
  bears/borneBy   independent continuant ↔ specifically dependent continuant  [BFO_0000053/…]
  inheres_in      SDC → independent continuant                                 [BFO_0000052]
  participates_in continuant → process                                         [BFO_0000056]
  is_about/designates/prescribes   ICE (GDC) → entity                          [IAO/CCO]
  has_part        same-category (continuant↔continuant, occurrent↔occurrent)   [BFO_0000051]
  abstracts       NO accepted counterpart → ungrounded coinage, always flagged

Subject category = the head's declared BFO anchor (or its CCO anchor walked up to BFO via the merged
grounding backbone). Object category = the filler slot's head-anchor when the filler is itself a
catalog head; else unverifiable (reported, not guessed). NAMESPACE DOCTRINE (RH): we never mint in
bfo:/cco: — remediation grounds sdg: relations via rdfs:subPropertyOf to the REAL BFO IRIs + signatures,
then re-authors the axioms that fail. Output: worklist → build/relation_lint.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# BFO numeral → category bucket (BFO-2020)
_BUCKET = {}
for n in ("0000004", "0000040", "0000030", "0000027", "0000024", "0000029", "0000006",
          "0000140", "0000141", "0000018", "0000026", "0000028", "0000009"):
    _BUCKET[n] = "independent"
for n in ("0000017", "0000023", "0000016", "0000034"):
    _BUCKET[n] = "realizable"
_BUCKET["0000019"] = "quality"
_BUCKET["0000020"] = "sdc"
_BUCKET["0000031"] = "gdc"
for n in ("0000015", "0000003", "0000035", "0000008", "0000038", "0000148", "0000182", "0000011"):
    _BUCKET[n] = "occurrent"

_CONT = {"independent", "realizable", "quality", "sdc", "gdc"}
SIG: "dict[str, dict | str | None]" = {
    "realizes":        {"domain": {"occurrent"}, "range": {"realizable"}},
    "abstracts":       None,                                     # ungrounded coinage — always flag
    "bears":           {"domain": {"independent"}, "range": {"realizable", "quality", "sdc"}},
    "borneBy":         {"domain": {"realizable", "quality", "sdc"}, "range": {"independent"}},
    "inheres_in":      {"domain": {"realizable", "quality", "sdc"}, "range": {"independent"}},
    "participates_in": {"domain": _CONT, "range": {"occurrent"}},
    "is_about":        {"domain": {"gdc"}, "range": None},
    "designates":      {"domain": {"gdc"}, "range": None},
    "prescribes":      {"domain": {"gdc"}, "range": None},
    "has_part":        "same",                                    # same-bucket-family check
}

_HEAD = re.compile(r"Class:\s*\{(\w+):Class\}\s*SubClassOf:\s*([a-z]+:[\w]+)")
_REST = re.compile(r"sdg:(\w+)\s+(?:some|only|min \d+|max \d+|exactly \d+)\s+\{(\w+):Class\}")


def _cco_to_bucket() -> "dict[str, str]":
    """cco IRI-local → bucket, walked up the merged grounding backbone to a BFO numeral."""
    ttl = REPO / "build" / "grounding" / "cco-merged.ttl"
    out: "dict[str, str]" = {}
    if not ttl.exists():
        return out
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(ttl)
        RDFS = rdflib.RDFS
        parent = {s: o for s, o in g.subject_objects(RDFS.subClassOf)
                  if isinstance(o, rdflib.URIRef)}
        for s in set(parent):
            cur, seen = s, set()
            while cur in parent and cur not in seen:
                seen.add(cur)
                cur = parent[cur]
                m = re.search(r"BFO_(\d{7})$", str(cur))
                if m and m.group(1) in _BUCKET:
                    out[str(s).rsplit("/", 1)[-1].rsplit("#", 1)[-1]] = _BUCKET[m.group(1)]
                    break
    except Exception as e:  # noqa: BLE001
        print(f"  (cco backbone unavailable: {str(e)[:80]})")
    return out


def bucket_of(anchor: str, cco: "dict[str, str]") -> "str | None":
    m = re.match(r"bfo:(\d{7})", anchor)
    if m:
        return _BUCKET.get(m.group(1))
    if anchor.startswith("cco:"):
        return cco.get(anchor.split(":", 1)[1])
    return None


def main() -> int:
    cat = json.loads((REPO / "src/aegir/ontology/catalog/catalog.json").read_text())
    tpls = cat.get("templates", cat if isinstance(cat, list) else [])
    cco = _cco_to_bucket()
    heads: "dict[str, str]" = {}
    parsed = []
    for t in tpls:
        mt = t.get("manchester_template") or ""
        hm = _HEAD.search(mt)
        if not hm:
            continue
        head, anchor = hm.group(1), hm.group(2)
        heads[head] = anchor
        parsed.append((t, head, anchor, _REST.findall(mt)))

    viol, unverifiable = [], 0
    rel_use = Counter()
    for t, head, anchor, rests in parsed:
        sb = bucket_of(anchor, cco)
        for rel, filler in rests:
            rel_use[rel] += 1
            sig = SIG.get(rel)
            fb = bucket_of(heads.get(filler, ""), cco) if filler in heads else None
            if rel not in SIG:
                continue                                   # not in the audited set
            if sig is None:
                viol.append({"head": head, "anchor": anchor, "relation": rel, "filler": filler,
                             "kind": "UNGROUNDED_RELATION",
                             "why": f"sdg:{rel} has no accepted BFO/RO/IAO counterpart to ground to"})
                continue
            if sig == "same":
                if sb and fb and ((sb in _CONT) != (fb in _CONT)):
                    viol.append({"head": head, "anchor": anchor, "relation": rel, "filler": filler,
                                 "kind": "CROSS_CATEGORY_PARTHOOD",
                                 "why": f"has_part crosses continuant/occurrent ({sb} → {fb})"})
                continue
            if sb and sb not in sig["domain"]:
                viol.append({"head": head, "anchor": anchor, "relation": rel, "filler": filler,
                             "kind": "DOMAIN_MISCAST",
                             "why": f"subject is {sb} ({anchor}); {rel} needs {'/'.join(sorted(sig['domain']))}"})
            elif sig["range"] and fb and fb not in sig["range"]:
                viol.append({"head": head, "anchor": anchor, "relation": rel, "filler": filler,
                             "kind": "RANGE_MISCAST",
                             "why": f"filler {filler} is {fb}; {rel} needs {'/'.join(sorted(sig['range']))}"})
            elif sig["range"] and fb is None:
                unverifiable += 1

    print(f"{len(parsed)} templates · relation use in audited set: {dict(rel_use.most_common())}")
    print(f"cco backbone resolved: {len(cco)} classes\n")
    kinds = Counter(v["kind"] for v in viol)
    print(f"VIOLATIONS: {len(viol)}  {dict(kinds)}  · range-unverifiable (filler not a head): {unverifiable}")
    for v in viol[:14]:
        print(f"  · {v['head']:38s} {v['kind']:24s} {v['why'][:80]}")
    if len(viol) > 14:
        print(f"  … {len(viol) - 14} more")
    out = REPO / "build" / "relation_lint.json"
    out.write_text(json.dumps({"violations": viol, "relation_use": dict(rel_use),
                               "unverifiable_range": unverifiable}, indent=1))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
