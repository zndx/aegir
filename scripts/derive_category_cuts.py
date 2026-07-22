#!/usr/bin/env python
"""derive_category_cuts — justification-driven tier-1 repairs for the union TBox core.

The taxonomy-core probe isolates the ~42-class contradiction core under the 1,836-class
propagation cascade; explain() names each core class's minimal axiom set. The mechanical
cut rule (validated on CourtDocument / FinancialReport / Population):

    a bare asserted SubClassOf(X, Y) is CUT iff X carries an EquivalentClasses
    ObjectIntersectionOf identity whose GENUS g and Y lift into DISJOINT category
    branches (identity wins over accretion — the ≡ is the class's certified
    definition; the bare edge is the contradicting accretion).

Category lift = ancestor closure over the union's bare taxonomy + the imported
cco-module; disjointness = the doc's DisjointClasses lines ∪ the justification-operative
BFO pairs. Classes with no ≡ identity (no genus to adjudicate by) are ESCALATED, never
guessed. → build/category_cuts.json, consumed by realize_sdg --reconcile-categories.

    uv run python scripts/derive_category_cuts.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

JUST = REPO / "build/taxcore_justifications.json"
UNION = REPO / "build/unified_realize/sdg-ontology.omn"
CCO_MODULE = REPO / "build/grounding/cco-module.ttl"
OUT = REPO / "build/category_cuts.json"

# BFO pairs the justifications actually exercise (plus the doc's own DisjointClasses).
BASE_DISJOINT = [("bfo:0000002", "bfo:0000003"), ("bfo:0000004", "bfo:0000031"),
                 ("bfo:0000020", "bfo:0000031"), ("bfo:0000004", "bfo:0000020")]

_IRI2PFX = [
    (re.compile(r"https://signals\.zndx\.org/sdg#([\w.-]+)"), r"sdg:\1"),
    (re.compile(r"http://purl\.obolibrary\.org/obo/BFO_(\d{7})"), r"bfo:\1"),
    (re.compile(r"https://www\.commoncoreontologies\.org/(ont\d+)"), r"cco:\1"),
]


def norm(t: str) -> str:
    for rx, rep in _IRI2PFX:
        t = rx.sub(rep, t)
    return t.strip("<>")


def short(a: str) -> str:
    """Functional-syntax token → prefixed form (justification strings carry bare locals)."""
    a = norm(a)
    a = re.sub(r"(?<![\w:])BFO_(\d{7})", r"bfo:\1", a)
    a = re.sub(r"(?<![\w:])(ont\d{8})", r"cco:\1", a)
    return a


def doc_taxonomy(omn: str) -> "tuple[dict, dict]":
    """(parents, edge_origins) from BOTH frame shapes. Bare top-level items are parents;
    bare conjuncts of top-level ``and``-chains are parents too (the ≡-genus chain runs
    through them). edge_origins[(cls, parent)] ⊆ {catalog, entity} — full-IRI headers are
    the certified catalog side."""
    bare = re.compile(r"^(?:bfo:\d{7}|cco:\w+|sdg:[\w.-]+)$")
    sect = re.compile(r"^\s*(?:SubClassOf|EquivalentTo):\s*(.*)$")
    frame = re.compile(r"^Class:\s*(<[^>\n]+>|(?:sdg|bfo|cco):[\w.-]+)([^\n]*)\n"
                       r"((?:    [^\n]*\n)*)", re.M)
    parents: "dict[str, set]" = {}
    origins: "dict[tuple, set]" = {}
    for m in frame.finditer(omn):
        cls = norm(m.group(1))
        origin = "catalog" if m.group(1).startswith("<") else "entity"
        for line in ([m.group(2)] if m.group(2).strip() else []) + m.group(3).splitlines():
            sm = sect.match(line)
            if not sm:
                continue
            depth, instr, cur, items = 0, False, "", []
            rest = norm(sm.group(1))
            for i, ch in enumerate(rest):
                if ch == '"' and (i == 0 or rest[i - 1] != "\\"):
                    instr = not instr
                elif not instr:
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
            for it in items:
                it = it.strip()
                cands = [it] if bare.match(it) else (
                    [c.strip().strip("()") for c in re.split(r"\s+and\s+", it)]
                    if " and " in it else [])
                for c in cands:
                    if bare.match(c) and c != cls:
                        parents.setdefault(cls, set()).add(c)
                        origins.setdefault((cls, c), set()).add(origin)
    return parents, origins


def module_taxonomy() -> "dict[str, set]":
    import rdflib
    g = rdflib.Graph()
    g.parse(str(CCO_MODULE))
    RDFS = rdflib.RDFS
    parents: "dict[str, set]" = {}
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            parents.setdefault(norm(str(s)), set()).add(norm(str(o)))
    return parents


def main() -> int:
    why = json.loads(JUST.read_text())["why"]
    omn = UNION.read_text()
    parents, edge_origins = doc_taxonomy(omn)
    for c, ps in module_taxonomy().items():
        parents.setdefault(c, set()).update(ps)

    disjoint = {frozenset(p) for p in BASE_DISJOINT}
    for m in re.finditer(r"^DisjointClasses:\s*([^\n]+)$", omn, flags=re.M):
        atoms = [norm(x.strip()) for x in m.group(1).split(",")]
        if len(atoms) == 2 and all(re.match(r"^\w+:[\w.-]+$", a) for a in atoms):
            disjoint.add(frozenset(atoms))

    def ancestors(c: str, seen=None) -> "set[str]":
        seen = seen if seen is not None else set()
        if c in seen:
            return set()
        seen.add(c)
        out = {c}
        for p in parents.get(c, ()):
            out |= ancestors(p, seen)
        return out

    def conflicting(a: str, b: str) -> "str | None":
        aa, bb = ancestors(a), ancestors(b)
        for pair in disjoint:
            x, y = tuple(pair)
            if (x in aa and y in bb) or (y in aa and x in bb):
                return f"{x} ⊥ {y}"
        return None

    genus_rx = re.compile(r"EquivalentClasses\((\S+) ObjectIntersectionOf\((\S+)[ )]")
    sub_rx = re.compile(r"SubClassOf\((\S+) (\S+)\)$")

    cuts, escalations = [], []
    for iri, j in why.items():
        cls = short(iri.rsplit("#", 1)[-1])
        cls = cls if ":" in cls else f"sdg:{cls}"
        axioms = [short(a) for a in j["axioms"]]
        # identity genus for every named class in this justification
        genus = {}
        for a in axioms:
            gm = genus_rx.match(a)
            if gm:
                g_cls = gm.group(1) if ":" in gm.group(1) else f"sdg:{gm.group(1)}"
                g_gen = gm.group(2) if ":" in gm.group(2) else f"sdg:{gm.group(2)}"
                genus[g_cls] = g_gen
        found = False
        for a in axioms:
            sm = sub_rx.match(a)
            if not sm:
                continue
            x = sm.group(1) if ":" in sm.group(1) else f"sdg:{sm.group(1)}"
            y = sm.group(2) if ":" in sm.group(2) else f"sdg:{sm.group(2)}"
            if not x.startswith("sdg:"):
                continue                          # imported truth is never cut
            g = genus.get(x)
            if not g:
                continue
            via = conflicting(g, y)
            if via:
                cuts.append({"class": x, "cut_parent": y, "genus": g, "via": via,
                             "unsat_witness": cls})
                found = True
        if not found:
            # certificate authority: a no-≡ class asserting bare edges on BOTH sides of a
            # disjointness, where exactly one side is catalog-asserted → cut the
            # non-catalog edge. Same-origin conflicts stay escalated.
            subs = [(sm.group(1) if ":" in sm.group(1) else f"sdg:{sm.group(1)}",
                     sm.group(2) if ":" in sm.group(2) else f"sdg:{sm.group(2)}")
                    for sm in (sub_rx.match(a) for a in axioms) if sm]
            for x, y1 in subs:
                if found or not x.startswith("sdg:"):
                    continue
                for x2, y2 in subs:
                    if x2 != x or y2 == y1:
                        continue
                    via = conflicting(y1, y2)
                    if not via:
                        continue
                    o1 = edge_origins.get((x, y1), set())
                    o2 = edge_origins.get((x, y2), set())
                    gone = None
                    if o1 == {"catalog"} and "catalog" not in o2:
                        gone = y2
                    elif o2 == {"catalog"} and "catalog" not in o1:
                        gone = y1
                    if gone:
                        cuts.append({"class": x, "cut_parent": gone,
                                     "genus": None, "via": via,
                                     "rule": "certificate_authority",
                                     "unsat_witness": cls})
                        found = True
                        break
        if not found:
            # rule 3 — cross-subject certificate pairs: two bare edges on DIFFERENT sdg
            # subjects whose targets conflict (functional-property intersections, chained
            # restriction forcing). Cut only when exactly ONE entity-origin candidate
            # exists against catalog counterparts — ambiguity escalates.
            pool = [(x, y) for x, y in subs if x.startswith("sdg:")]
            cands = set()
            for i_, (x1, y1) in enumerate(pool):
                for x2, y2 in pool[i_ + 1:]:
                    if (x1, y1) == (x2, y2):
                        continue
                    if conflicting(y1, y2):
                        for e_ in ((x1, y1), (x2, y2)):
                            o_ = edge_origins.get(e_, set())
                            if o_ and "catalog" not in o_:
                                cands.add(e_)
            if len(cands) == 1:
                x, gone = next(iter(cands))
                cuts.append({"class": x, "cut_parent": gone, "genus": None,
                             "via": "cross-subject pair", "rule": "certificate_pair",
                             "unsat_witness": cls})
                found = True
        if not found:
            escalations.append({"class": cls,
                                "reason": "no ≡-genus-adjudicable bare edge in justification",
                                "axioms": axioms})

    # dedup (same cut witnessed by many core classes) + MERGE with prior rounds —
    # the repair loop is iterative (probe core → explain → cut → re-probe until dry).
    prior = []
    if OUT.exists():
        prior = json.loads(OUT.read_text()).get("cuts", [])
    seen, uniq = set(), []
    for c in prior + cuts:
        k = (c["class"], c["cut_parent"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    n_new = len(uniq) - len(prior)
    print(f"merged: {len(prior)} prior + {n_new} new")
    OUT.write_text(json.dumps({"cuts": uniq, "escalations": escalations}, indent=1))
    print(f"cuts: {len(uniq)} unique edges (from {len(cuts)} witnesses) · "
          f"escalations: {len(escalations)}")
    for c in uniq[:12]:
        print(f"  CUT {c['class']} ⊑ {c['cut_parent']}  (genus {c['genus']}; {c['via']})")
    for e in escalations[:5]:
        print(f"  ESC {e['class']}: {e['reason']}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
