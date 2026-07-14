"""discriminability — a metrology observable + the SHACL-arc lead-in.

For Data Element *elucidation* (recover a column/relational-construct's ontology identity from its projection),
the question is: is each Data Element's identity UNIQUELY DETERMINED by its observable relational projection,
DISTINCT from its near-neighbours? Two sibling classes are CONFUSABLE when their SHACL shapes carry no
observable DISTINGUISHING FEATURE (a differing datatype / enum / cardinality / object-property target, or a
property present in one and not the other). A confusable pair routes by its ontology status:

  - DISJOINT in OWL but confusable in the projection  → PROJECTION GAP  (the differentia was dropped by the
    shape/DDL projection) → fix in the SHACL arc: carry the differentia into the shape. [the #16 backlog]
  - NOT distinguished in OWL                          → ONTOLOGY GAP    (no observable differentia at all)
    → fix by ontology refinement: add the identity-constituting property (or mark the pair non-observable).

Guard (observable ≠ logical identity): a collapsed projection is a DEFECT only when a data-observable
differentia exists to preserve; a purely metaphysical distinction with no data footprint is not a projection
defect. So the trichotomy is: differentia in-shape → discriminable; differentia in-OWL-not-shape → projection
gap; no differentia → ontology gap (add one, or record legitimately-non-observable).

One computation, four uses: (1) a gateable metrology observable, (2) the SHACL-arc worklist (projection gaps),
(3) the ontology-refinement worklist (confusable-and-undistinguished), (4) the DEE eval's difficulty labeller +
distractor set — confusable/near-confusable pairs are the hard rungs; the distinguishing feature is the
required evidence; the hop-count to it is the difficulty tier (in-cell = L0, one object-property = L1, …).

Deterministic; no LLM / GPU / network. Inputs: shapes.ttl (SHACL projection) + the ontology (hierarchy →
siblings) + optional adjudications.json (declared disjointness). [[ontology_metrology_iof]] [[sibling adjudication]]
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import rdflib

SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")


def _local(iri) -> str:
    return str(iri).split("#")[-1].split("/")[-1].split(":")[-1]  # strip URI frag/path AND CURIE prefix


def _sig(proj: "dict[str, tuple]") -> tuple:
    """The OBSERVABLE type-signature of a class — the multiset of constraint descriptors with the bespoke
    property NAMES abstracted away. Two classes with the same signature are confusable at the type level: a
    model can't tell them apart from column types alone (the `9.8` case: {decimal, string} == {decimal, string}
    unless a disambiguating enum or object-property-target breaks the tie). Enums carry their VALUE set and
    object-properties carry their TARGET class, since those genuinely discriminate."""
    desc = []
    for feat in proj.values():
        kind, val = feat
        if kind == "enum":
            desc.append("enum:" + "|".join(sorted(val)))     # the value set discriminates
        elif kind == "ref":
            desc.append("ref:" + str(val))                    # the join target discriminates
        else:
            desc.append(str(val or "any"))                    # xsd datatype
    return tuple(sorted(desc))


# ── the observable projection of each class, from its SHACL shape ──────────────────────────────────────────
def shape_projections(shapes_ttl: Path) -> "dict[str, dict[str, tuple]]":
    """{class_local: {path_local: feature}} where feature is a hashable observable signature:
    ('dt', datatype)              — a datatype column (in-cell, tier L0)
    ('enum', frozenset(values))   — a closed value set (the sharpest in-cell discriminator, L0)
    ('ref', target_class)         — an object property = an FK join (multi-table, tier L1+)."""
    g = rdflib.Graph().parse(str(shapes_ttl), format="turtle")
    out: "dict[str, dict[str, tuple]]" = {}
    for s in g.subjects(rdflib.RDF.type, SH.NodeShape):
        tc = next(iter(g.objects(s, SH.targetClass)), None)
        if tc is None:
            continue
        proj: "dict[str, tuple]" = {}
        for p in g.objects(s, SH.property):
            path = next(iter(g.objects(p, SH["path"])), None)
            if path is None:
                continue
            key = _local(path)
            enum = [o for o in g.objects(p, SH["in"])]
            if enum:  # sh:in is an RDF list head; collect members
                vals = frozenset(str(x) for x in g.items(enum[0]))
                proj[key] = ("enum", vals)
            elif (cl := next(iter(g.objects(p, SH["class"])), None)) is not None:
                proj[key] = ("ref", _local(cl))
            elif (dt := next(iter(g.objects(p, SH.datatype)), None)) is not None:
                proj[key] = ("dt", _local(dt))
            else:
                proj[key] = ("any", None)
        out[_local(tc)] = proj
    return out


# ── the class hierarchy → sibling near-neighbours (Manchester omn: Class: / SubClassOf: named parents) ──────
def sibling_pairs(omn_path: Path) -> "tuple[list[tuple[str, str]], dict[str, list[str]]]":
    """Named-class SubClassOf only (skip restrictions) → parent→children → sibling pairs."""
    text = omn_path.read_text()
    kids: "dict[str, set[str]]" = defaultdict(set)
    # split into Class frames
    for m in re.finditer(r"^Class:\s+(\S+)(.*?)(?=^Class:|\Z)", text, re.M | re.S):
        cls = _local(m.group(1))
        sc = re.search(r"^\s*SubClassOf:\s*(.*?)(?=^\s*[A-Z][A-Za-z]+:|\Z)", m.group(2), re.M | re.S)
        if not sc:
            continue
        for ref in sc.group(1).split(","):
            ref = ref.strip()
            # a NAMED parent is a bare token (sdg:X / bfo:X / <IRI>) with no restriction keyword
            if ref and not re.search(r"\b(some|only|value|min|max|exactly|and|or|not)\b", ref) and " " not in ref.strip("<>"):
                kids[_local(ref)].add(cls)
    groups = {p: sorted(c) for p, c in kids.items() if len(c) >= 2}
    pairs = [(a, b) for c in groups.values() for i, a in enumerate(c) for b in c[i + 1:]]
    return pairs, groups


# ── disjointness (optional; aligns the sharpest signal) ────────────────────────────────────────────────────
def disjoint_pairs(adjudications: Path) -> "set[frozenset]":
    try:
        adj = json.loads(adjudications.read_text())
    except (OSError, ValueError):
        return set()
    out: "set[frozenset]" = set()
    for fam in (adj.get("families") or {}).values():
        for pr in fam.get("disjoint_pairs", []):
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                out.add(frozenset(_local(x) for x in pr))
        if fam.get("default") == "disjoint":
            mem = [_local(x) for x in fam.get("members", [])]
            for i, a in enumerate(mem):
                for b in mem[i + 1:]:
                    out.add(frozenset((a, b)))
    return out


_TIER = {"enum": 0, "dt": 0, "any": 0, "ref": 1}  # in-cell = L0; object-property/join = L1 (chains → higher, later)


def distinguishing_features(pa: dict, pb: dict) -> "list[tuple[str, str, int]]":
    """(path, reason, tier) where the two projections observably differ."""
    feats = []
    for path in set(pa) | set(pb):
        fa, fb = pa.get(path), pb.get(path)
        if fa == fb:
            continue
        if fa is None or fb is None:
            present = fa or fb
            feats.append((path, "present-in-one", _TIER.get(present[0], 0)))
        elif fa[0] != fb[0]:
            feats.append((path, f"kind {fa[0]}≠{fb[0]}", min(_TIER.get(fa[0], 0), _TIER.get(fb[0], 0))))
        else:
            feats.append((path, f"{fa[0]} value differs", _TIER.get(fa[0], 0)))
    return sorted(feats, key=lambda f: f[2])


def compute(shapes_ttl: Path, omn_path: Path, adjudications: "Path | None" = None) -> dict:
    proj = shape_projections(shapes_ttl)
    disj = disjoint_pairs(adjudications) if adjudications else set()
    _pairs, groups = sibling_pairs(omn_path)
    # STRUCTURAL adequacy of the ontology for elucidation (the two prerequisites the metric needs):
    non_bfo_parents = [p for p in groups if re.match(r"^[A-Z][a-z]", p)]  # a CamelCase sdg: class, not a BFO/CCO code
    structural = {
        "disjointness_axioms": len(disj),                          # 0 ⇒ no distinction ground truth
        "taxonomy_parents_total": len(groups),                     # how many classes serve as parents
        "taxonomy_parents_domain": len(non_bfo_parents),           # domain (non-BFO/CCO) mid-tier — the near-neighbour source
        "hierarchy": "flat (classes anchor directly to BFO/CCO)" if len(non_bfo_parents) < 5
                     else "has domain mid-tier",
    }
    # OBSERVABLE confusability by type-signature (names abstracted) — computable now, honest on a flat ontology
    buckets: "dict[tuple, list[str]]" = defaultdict(list)
    for cls, pr in proj.items():
        buckets[_sig(pr)].append(cls)
    clusters = sorted((cs for cs in buckets.values() if len(cs) >= 2), key=len, reverse=True)
    in_collision = sum(len(cs) for cs in clusters)
    # route each collision cluster: any declared-disjoint member pair ⇒ projection gap; else ⇒ ontology/undistinguished
    proj_gap_clusters = [cs for cs in clusters
                         if any(frozenset((a, b)) in disj for i, a in enumerate(cs) for b in cs[i + 1:])]
    n = len(proj) or 1
    return {
        "n_classes_with_shape": len(proj),
        "structural_adequacy": structural,
        "discriminability_typesig": round(1 - in_collision / n, 4),   # 1 = every class has a unique observable type-sig
        "confusable_clusters": len(clusters),
        "classes_in_collision": in_collision,
        "largest_clusters": [{"n": len(cs), "classes": cs[:8], "signature": list(_sig(proj[cs[0]]))[:10]}
                             for cs in clusters[:15]],
        "projection_gap_clusters": len(proj_gap_clusters),            # collisions that OWL says are disjoint → SHACL arc
        "note": "type-signature collision is a LOWER BOUND on confusability (names abstracted, values/enums kept); "
                "real Data-Element discriminability needs the SKOS Data-Element mapping + disjointness — the SHACL arc.",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--ontology", required=True, help="the realized ontology .omn (hierarchy → siblings)")
    ap.add_argument("--adjudications", default="src/aegir/ontology/catalog/adjudications.json")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    adj = Path(a.adjudications) if a.adjudications and Path(a.adjudications).exists() else None
    m = compute(Path(a.shapes), Path(a.ontology), adj)
    s = m["structural_adequacy"]
    print(f"discriminability (type-signature) {m['discriminability_typesig']}  over {m['n_classes_with_shape']} classes")
    print(f"  confusable clusters: {m['confusable_clusters']}  · classes in collision: {m['classes_in_collision']}"
          f"  · projection-gap clusters (disjoint-yet-collided): {m['projection_gap_clusters']}")
    print(f"  STRUCTURAL adequacy: disjointness axioms={s['disjointness_axioms']} · "
          f"domain taxonomy parents={s['taxonomy_parents_domain']} · hierarchy={s['hierarchy']}")
    if m["largest_clusters"]:
        c = m["largest_clusters"][0]
        print(f"  largest collision ({c['n']} classes): {c['classes'][:5]} … sig={c['signature'][:6]}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(m, indent=1, default=str))
        print(f"→ {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
