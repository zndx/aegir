#!/usr/bin/env python
"""derive_property_domains — CREATE rdfs:domain/range for the sdg vocabulary (RH 2026-07-21).

The meta-structural core: neither generation authors rdfs:domain on derived sdg: properties
(catalog carries only the 11 BFO signature-scaffold domains; the run none). This instrument
DERIVES them from usage — for every property, the anchor-lifted histogram of its restriction
SUBJECTS (domain side) and object FILLERS (range side) across BOTH realizations — and emits:

  * build/domain_candidates.json — per-property profiles, shares, banded verdicts
  * src/aegir/ontology/property_domains.py — DOMAINS_OMN frames for the AUTO band, staged
    behind AEGIR_PROPERTY_DOMAINS (default OFF) exactly like the #27 signature arming

Registered boundary conditions (engineering tools, fixed BEFORE measurement):
  dominant-anchor share ≥ 0.70 → AUTO candidate (the emitted leg)
  0.40 ≤ share < 0.70         → REVIEW (CAS worklist — never dropped)
  share < 0.40                → POLYSEMOUS (property-split candidate)
  object+data mixed usage     → PUNNED (review; no emission)

Anchors lift to the FIRST bfo:/cco: ancestor (specific cco beats generic bfo — the
grounding-debt doctrine); malformed anchors (e.g. 8-digit bfo:00000995) are surfaced.

EXPANSION PATH (RH 2026-07-21): harvest sources are pluggable — FIBO integration and PRODML
incorporation (docs.energistics.org/PRODML) add sources through the same census/fragment
machinery; the property universe grows, bands recompute, and every expansion re-enters the
verify→arm discipline before its domains bite.

    uv run python scripts/derive_property_domains.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_OMN = "/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06/ontology/sdg-ontology.omn"
AUTO_SHARE = 0.70          # registered pre-measurement
REVIEW_SHARE = 0.40

_SDG = "https://signals.zndx.org/sdg#"
_BFO = "http://purl.obolibrary.org/obo/BFO_"
_CCO = "https://www.commoncoreontologies.org/"


def _curie(iri: str) -> str:
    if iri.startswith(_SDG):
        return "sdg:" + iri[len(_SDG):]
    if iri.startswith(_BFO):
        return "bfo:" + iri[len(_BFO):]
    if iri.startswith(_CCO):
        return "cco:" + iri[len(_CCO):]
    return iri


def harvest_catalog(usage, parents, declared):
    from rdflib import BNode, Graph, URIRef
    from rdflib.namespace import OWL, RDF, RDFS
    g = Graph()
    g.parse(str(REPO / "corpora/ontology/sdg-ontology.owl"))

    def first(s_, p_):
        for o in g.objects(s_, p_):
            return o
        return None

    for p_ in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(p_, URIRef):
            declared[_curie(str(p_))] = "ObjectProperty"
    for p_ in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(p_, URIRef):
            declared[_curie(str(p_))] = "DataProperty"
    for s_ in g.subjects(RDF.type, OWL.Class):
        if not isinstance(s_, URIRef):
            continue
        subj = _curie(str(s_))
        for o in g.objects(s_, RDFS.subClassOf):
            if isinstance(o, URIRef):
                parents[subj].add(_curie(str(o)))
        for o in list(g.objects(s_, RDFS.subClassOf)) + list(g.objects(s_, OWL.equivalentClass)):
            todo = [o]
            while todo:
                e = todo.pop()
                if not isinstance(e, BNode):
                    continue
                lst = first(e, OWL.intersectionOf)
                if lst is not None:
                    while lst and lst != RDF.nil:
                        todo.append(first(lst, RDF.first))
                        lst = first(lst, RDF.rest)
                    continue
                prop = first(e, OWL.onProperty)
                if not isinstance(prop, URIRef):
                    continue
                pc = _curie(str(prop))
                if not pc.startswith("sdg:"):
                    continue
                f = (first(e, OWL.someValuesFrom) or first(e, OWL.allValuesFrom)
                     or first(e, OWL.onClass))
                u = usage[pc]
                u["subjects"].append(subj)
                u["gens"].add("catalog")
                if isinstance(f, URIRef):
                    fc = _curie(str(f))
                    if fc.startswith("xsd:") or "XMLSchema" in str(f):
                        u["data_ranges"].append(str(f).rsplit("#", 1)[-1])
                    else:
                        u["fillers"].append(fc)


_RUN_REST = re.compile(r"(sdg:\w+)\s+(?:some|only|exactly|min|max)\s*\d*\s+"
                       r"(sdg:\w+|xsd:\w+|cco:\w+|bfo:\d+)")


def harvest_run(usage, parents, declared):
    txt = Path(RUN_OMN).read_text()
    for m in re.finditer(r"^(ObjectProperty|DataProperty):\s*(sdg:\w+)", txt, re.M):
        declared.setdefault(m.group(2), m.group(1))
    for m in re.finditer(r"^Class:\s*(sdg:\w+)\n((?:    .*\n)*)", txt, re.M):
        subj, body = m.group(1), m.group(2)
        for pm in re.finditer(r"(?:SubClassOf|EquivalentTo):\s*([^\n]*)", body):
            for atom in re.finditer(r"(?<![\w:])((?:bfo|cco|sdg):[\w]+)(?:\s*[,\n])", pm.group(1) + "\n"):
                a = atom.group(1)
                if not _RUN_REST.match(pm.group(1)[max(0, atom.start() - 30):]):
                    parents[subj].add(a)
        for rm in _RUN_REST.finditer(body):
            pc, filler = rm.groups()
            u = usage[pc]
            u["subjects"].append(subj)
            u["gens"].add("run")
            if filler.startswith("xsd:"):
                u["data_ranges"].append(filler[4:])
            else:
                u["fillers"].append(filler)


def make_lift(parents):
    memo = {}

    def lift(c: str, seen=frozenset()) -> str:
        if c in memo:
            return memo[c]
        if re.fullmatch(r"bfo:\d{7}", c) or c.startswith("cco:"):
            return c
        if re.fullmatch(r"bfo:\d+", c):
            return f"malformed:{c}"
        if c in seen:
            return "unanchored"
        r = "unanchored"
        for p_ in sorted(parents.get(c, ())):
            r2 = lift(p_, seen | {c})
            if r2 != "unanchored":
                r = r2
                break
        memo[c] = r
        return r

    return lift


def band(hist: Counter) -> "tuple[str, str | None, float]":
    total = sum(hist.values())
    if not total:
        return "unused", None, 0.0
    top, n = hist.most_common(1)[0]
    share = n / total
    if top.startswith(("malformed:", "unanchored")):
        return "review", top, round(share, 3)
    if share >= AUTO_SHARE:
        return "auto", top, round(share, 3)
    if share >= REVIEW_SHARE:
        return "review", top, round(share, 3)
    return "polysemous", top, round(share, 3)


def main() -> int:
    usage: "dict[str, dict]" = defaultdict(lambda: {"subjects": [], "fillers": [],
                                                    "data_ranges": [], "gens": set()})
    parents: "dict[str, set]" = defaultdict(set)
    declared: "dict[str, str]" = {}
    harvest_catalog(usage, parents, declared)
    harvest_run(usage, parents, declared)
    lift = make_lift(parents)

    rows = []
    omn_frames = []
    verdicts = Counter()
    for pc, u in sorted(usage.items()):
        dom_hist = Counter(lift(s_) for s_ in u["subjects"])
        rng_hist = Counter(lift(f_) for f_ in u["fillers"])
        punned = bool(u["fillers"]) and bool(u["data_ranges"])
        dv, dtop, dshare = band(dom_hist)
        rv, rtop, rshare = band(rng_hist) if u["fillers"] else ("n/a", None, 0.0)
        if punned:
            dv = rv = "punned"
        verdicts[dv] += 1
        rows.append({"property": pc, "declared": declared.get(pc, "undeclared"),
                     "n_uses": len(u["subjects"]), "generations": sorted(u["gens"]),
                     "domain": {"hist": dict(dom_hist.most_common(6)), "verdict": dv,
                                "candidate": dtop, "share": dshare},
                     "range": {"hist": dict(rng_hist.most_common(6)), "verdict": rv,
                               "candidate": rtop, "share": rshare},
                     "data_ranges": dict(Counter(u["data_ranges"]).most_common(3)),
                     "punned": punned})
        if not punned and (dv == "auto" or rv == "auto"):
            kind = declared.get(pc, "ObjectProperty")
            legs = []
            if dv == "auto":
                legs.append(f"    Domain: {dtop}")
            if rv == "auto" and kind == "ObjectProperty":
                legs.append(f"    Range: {rtop}")
            if legs:
                omn_frames.append(f"{kind}: {pc}\n" + "\n".join(legs))

    out = {"registered": {"auto_share": AUTO_SHARE, "review_share": REVIEW_SHARE},
           "n_properties": len(rows), "verdicts": dict(verdicts),
           "n_omn_frames": len(omn_frames), "properties": rows}
    (REPO / "build/domain_candidates.json").write_text(json.dumps(out, indent=1))

    mod = ('"""property_domains — GENERATED rdfs:domain/range for sdg properties '
           "(derive_property_domains.py; RH 2026-07-21 meta-structure arc).\n\n"
           "Usage-derived, banded (auto ≥ 0.70 dominant-anchor share), staged behind\n"
           "AEGIR_PROPERTY_DOMAINS (default OFF) until verify_property_domains reaches 0 unsat —\n"
           "the #27 arming discipline. NEVER hand-edit; re-run the deriver.\n"
           '"""\n\n'
           'DOMAINS_OMN = """\n' + "\n".join(omn_frames) + '\n"""\n')
    (REPO / "src/aegir/ontology/property_domains.py").write_text(mod)

    print(f"{len(rows)} sdg properties · verdicts {dict(verdicts)}")
    print(f"AUTO frames emitted: {len(omn_frames)} → property_domains.py (staged OFF)")
    print("→ build/domain_candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
