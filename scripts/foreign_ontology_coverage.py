#!/usr/bin/env python
"""foreign_ontology_coverage — could our axiom-pattern library express THIS ontology? (RH 2026-07-20)

The generative-envelope question, measured: parse a foreign ontology (FIBO first) AND our own
catalog into ONE shape-signature space, then classify every logical axiom of the foreign ontology
as expressible / not-expressible by the currently defined patterns. "Expressible" is judged at the
FAMILY level — a signature normalizes a conjunction to the SET of distinct conjunct kinds (our
patterns generate conjunctions of restriction kinds; arity is free) — so the verdict measures
constructs, not counts.

Signatures (recursive, set-normalized):
    C                     named class          some(F) only(F) value  — object restrictions
    exact-q(F) min-q(F) max-q(F)               qualified cardinalities (filler kind F)
    card-u                                     UNqualified cardinality
    d-some(D) d-only(D) d-value d-card         data restrictions
    and{...} or{...} not(...) oneOf            boolean/enumeration
Axioms:  Sub[sig] Equiv[sig] GCI Disjoint(2|n) DisjointUnion HasKey
         RBox: subProp domain range inverse chain char(Functional...) equivProp dPropAxioms
         ABox: type(C) type(complex) opa dpa negative

Our side = catalog manchester DSL (parsed with the same grammar) + the realizer machinery
(Domain/Range/InverseOf/SubPropertyOf signatures, pairwise DisjointClasses, Individual/Types
ABox, annotations). Anything else is a BLOCKER, bucketed and ranked with examples.

    uv run python scripts/foreign_ontology_coverage.py --root ~/local/src/oss/fibo --tag fibo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rdflib import BNode, Graph, URIRef  # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS, SKOS  # noqa: E402

ANNOT_NS = ("http://www.w3.org/2004/02/skos/core#", "http://purl.org/dc/terms/",
            "https://www.omg.org/spec/Commons/AnnotationVocabulary/",
            "http://www.omg.org/techprocess/ab/SpecificationMetadata/",
            "https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/AnnotationVocabulary/")
ANNOT_P = {RDFS.label, RDFS.comment, RDFS.seeAlso, RDFS.isDefinedBy, OWL.versionInfo,
           OWL.versionIRI, OWL.priorVersion, OWL.backwardCompatibleWith, OWL.incompatibleWith,
           OWL.deprecated, URIRef("http://purl.org/dc/elements/1.1/license")}


# ── expression classifier (rdflib side) ───────────────────────────────────────

def _first(g: Graph, s, p):
    for o in g.objects(s, p):
        return o
    return None


def _members(g: Graph, lst) -> list:
    out = []
    while lst and lst != RDF.nil:
        out.append(_first(g, lst, RDF.first))
        lst = _first(g, lst, RDF.rest)
    return [x for x in out if x is not None]


def sig(g: Graph, e, depth: int = 0) -> str:
    if depth > 6:
        return "deep"
    if isinstance(e, URIRef):
        return "C"
    if not isinstance(e, BNode):
        return "lit"
    inter = _first(g, e, OWL.intersectionOf)
    if inter is not None:
        return "and{" + ",".join(sorted({sig(g, m, depth + 1) for m in _members(g, inter)})) + "}"
    uni = _first(g, e, OWL.unionOf)
    if uni is not None:
        return "or{" + ",".join(sorted({sig(g, m, depth + 1) for m in _members(g, uni)})) + "}"
    if _first(g, e, OWL.complementOf) is not None:
        return f"not({sig(g, _first(g, e, OWL.complementOf), depth + 1)})"
    if _first(g, e, OWL.oneOf) is not None:
        return "oneOf"
    if _first(g, e, OWL.inverseOf) is not None and _first(g, e, OWL.onProperty) is None:
        return "invProp"                                   # anonymous inverse in property position
    on_p = _first(g, e, OWL.onProperty)
    if on_p is not None:
        inv = isinstance(on_p, BNode)
        px = "inv:" if inv else ""
        for pred, name in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only")):
            f = _first(g, e, pred)
            if f is not None:
                fk = sig(g, f, depth + 1)
                kind = "d-" if fk == "C" and isinstance(f, URIRef) and _is_datatype(g, f) else ""
                return f"{px}{kind}{name}({fk})"
        if _first(g, e, OWL.hasValue) is not None:
            v = _first(g, e, OWL.hasValue)
            return f"{px}{'d-' if not isinstance(v, (URIRef, BNode)) else ''}value"
        oc = _first(g, e, OWL.onClass)
        odr = _first(g, e, OWL.onDataRange)
        for pred, name in ((OWL.qualifiedCardinality, "exact-q"),
                           (OWL.minQualifiedCardinality, "min-q"),
                           (OWL.maxQualifiedCardinality, "max-q")):
            if _first(g, e, pred) is not None:
                if odr is not None:
                    return f"{px}d-{name}"
                return f"{px}{name}({sig(g, oc, depth + 1) if oc is not None else 'C'})"
        for pred in (OWL.cardinality, OWL.minCardinality, OWL.maxCardinality):
            if _first(g, e, pred) is not None:
                return f"{px}card-u"
        if _first(g, e, URIRef(str(OWL) + "hasSelf")) is not None:
            return "hasSelf"
    return "expr?"


def _is_datatype(g: Graph, u: URIRef) -> bool:
    s = str(u)
    return s.startswith("http://www.w3.org/2001/XMLSchema#") or (u, RDF.type, RDFS.Datatype) in g


# ── catalog side: same grammar over our manchester DSL ────────────────────────

_CONJ_SPLIT = re.compile(r",\s*(?![^()]*\))")
_R = re.compile(r"^(?:inverse\s+)?[\w.-]+:[\w.-]+\s+(some|only|value|exactly|min|max)\s*(\d+)?\s+(.+)$")
_XSD = re.compile(r"^xsd:[\w]+$")


def _filler_sig(f: str) -> str:
    f = f.strip()
    if f.startswith("(") and " or " in f:
        return "or{C}"
    if _XSD.match(f):
        return "D"
    return "C"


def _kinds_of(s_: str) -> "set[str]":
    """Axiom sig → its conjunct-kind set (and{a,b}→{a,b}; atomic→{itself})."""
    inner = s_[s_.index("[") + 1:-1]
    if inner.startswith("and{"):
        return set(inner[4:-1].split(","))
    return {inner}


def catalog_sigs() -> "tuple[set, dict]":
    cat = json.loads((REPO / "src/aegir/ontology/catalog/catalog.json").read_text())
    tpls = cat.get("templates", cat if isinstance(cat, list) else [])
    sigs: set = set()
    by_pattern: "dict[str, set]" = defaultdict(set)
    for t in tpls:
        mt = (t.get("manchester_template") or "").replace("\n", " ")
        mt = re.sub(r"\{\w+:Class\}", "SLOTC", mt)
        pattern = t.get("provenance", {}).get("pattern") or "(uncategorized)"
        for kw, tag in (("SubClassOf:", "Sub"), ("EquivalentTo:", "Equiv")):
            if kw not in mt:
                continue
            body = mt.split(kw, 1)[1]
            for other in ("SubClassOf:", "EquivalentTo:", "DisjointWith:"):
                if other in body:
                    body = body.split(other, 1)[0]
            conjs = [c.strip() for c in _CONJ_SPLIT.split(body) if c.strip()]
            kinds: set = set()
            for c in conjs:
                c = c.strip().strip("()") if c.startswith("(") and c.endswith(")") and " or " not in c else c
                m = _R.match(c)
                if m:
                    q, _n, filler = m.groups()
                    fs = _filler_sig(filler)
                    if fs == "D":
                        kinds.add({"some": "d-some(C)", "only": "d-only(C)", "value": "d-value",
                                   "exactly": "d-exact-q", "min": "d-min-q", "max": "d-max-q"}[q])
                    else:
                        kinds.add({"some": f"some({fs})", "only": f"only({fs})", "value": "value",
                                   "exactly": f"exact-q({fs})", "min": f"min-q({fs})",
                                   "max": f"max-q({fs})"}[q])
                elif " or " in c:
                    kinds.add("or{C}")
                else:
                    kinds.add("C")
            s_ = f"{tag}[" + ("and{" + ",".join(sorted(kinds)) + "}" if len(kinds) > 1
                              else (next(iter(kinds)) if kinds else "C")) + "]"
            sigs.add(s_)
            by_pattern[pattern].add(s_)
    # the realizer machinery (frames it emits beyond the templates)
    machinery = {"rbox:subProp", "rbox:domain", "rbox:range", "rbox:inverse",
                 "Disjoint(2)", "abox:type(C)", "annotation"}
    sigs |= machinery
    for m in machinery:
        by_pattern["(realizer-machinery)"].add(m)
    return sigs, dict(by_pattern)


def _normalize_foreign(s_: str) -> str:
    """Foreign axiom sig → catalog-space: exact filler-kind retained; the family match is on the
    normalized string. Cardinality NUMBERS are already erased on both sides."""
    return s_


# ── foreign ontology sweep ────────────────────────────────────────────────────

def sweep(root: Path, skip_pat: str) -> "tuple[Counter, dict, Counter, dict]":
    files = [p for p in sorted(root.rglob("*.rdf")) if not re.search(skip_pat, p.name)]
    ax = Counter()
    examples: "dict[str, list]" = defaultdict(list)
    meta = Counter()
    per_module = Counter()
    for i, f in enumerate(files):
        g = Graph()
        try:
            g.parse(str(f))
        except Exception as e:  # noqa: BLE001
            meta["parse_errors"] += 1
            continue
        module = f.relative_to(root).parts[0]
        ann_props = {s for s in g.subjects(RDF.type, OWL.AnnotationProperty)}
        obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
        dat_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
        inds = set(g.subjects(RDF.type, OWL.NamedIndividual))

        def _rec(kind: str, subj):
            ax[kind] += 1
            per_module[module] += 1
            if len(examples[kind]) < 3:
                examples[kind].append(f"{module}: {str(subj).rsplit('/', 1)[-1][:70]}")

        for s, p, o in g:
            if p in ANNOT_P or p in ann_props or any(str(p).startswith(ns) for ns in ANNOT_NS):
                meta["annotations"] += 1
                continue
            if p == RDF.type:
                if o in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty,
                         OWL.NamedIndividual, OWL.Ontology, RDFS.Datatype, OWL.Restriction,
                         OWL.Axiom, OWL.AllDisjointClasses, OWL.AllDifferent):
                    meta["declarations"] += 1
                elif o in (OWL.FunctionalProperty, OWL.InverseFunctionalProperty, OWL.TransitiveProperty,
                           OWL.SymmetricProperty, OWL.AsymmetricProperty, OWL.ReflexiveProperty,
                           OWL.IrreflexiveProperty):
                    _rec("rbox:characteristic", s)
                elif isinstance(s, URIRef) and s in inds:
                    _rec("abox:type(C)" if isinstance(o, URIRef) else "abox:type(complex)", s)
                elif isinstance(s, BNode):
                    pass
                else:
                    meta["type_other"] += 1
            elif p == RDFS.subClassOf:
                if isinstance(s, BNode):
                    _rec("GCI", s)
                else:
                    _rec(f"Sub[{sig(g, o)}]", s)
            elif p == OWL.equivalentClass:
                if isinstance(s, URIRef) and _is_datatype(g, s):
                    _rec("datatypeDef", s)
                else:
                    _rec(f"Equiv[{sig(g, o)}]", s)
            elif p == OWL.disjointWith:
                _rec("Disjoint(2)", s)
            elif p == OWL.disjointUnionOf:
                _rec("DisjointUnion", s)
            elif p == OWL.members and (s, RDF.type, OWL.AllDisjointClasses) in g:
                _rec("Disjoint(n)", s)
            elif p == RDFS.subPropertyOf:
                _rec("rbox:subProp", s)
            elif p == RDFS.domain:
                _rec(f"rbox:domain[{sig(g, o)}]" if isinstance(o, BNode) else "rbox:domain", s)
            elif p == RDFS.range:
                _rec(f"rbox:range[{sig(g, o)}]" if isinstance(o, BNode) else "rbox:range", s)
            elif p == OWL.inverseOf and isinstance(s, URIRef):
                _rec("rbox:inverse", s)
            elif p == OWL.propertyChainAxiom:
                _rec("rbox:chain", s)
            elif p == OWL.equivalentProperty:
                _rec("rbox:equivProp", s)
            elif p == OWL.propertyDisjointWith:
                _rec("rbox:disjointProp", s)
            elif p == OWL.hasKey:
                _rec("HasKey", s)
            elif p in (OWL.sameAs, OWL.differentFrom):
                _rec("abox:identity", s)
            elif s in inds and (p in obj_props or (isinstance(o, URIRef) and p not in (OWL.imports,))):
                if p == OWL.imports:
                    meta["imports"] += 1
                elif p in dat_props or not isinstance(o, (URIRef, BNode)):
                    _rec("abox:dpa", s)
                else:
                    _rec("abox:opa", s)
            elif p == OWL.imports:
                meta["imports"] += 1
            elif p == OWL.onDatatype or p == OWL.withRestrictions:
                _rec("datatypeFacet", s)
            elif isinstance(s, BNode):
                pass                                        # structure triples of expressions
            else:
                meta["other_triples"] += 1
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1}/{len(files)} files", flush=True)
    return ax, dict(examples), meta, per_module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="foreign")
    ap.add_argument("--skip", default=r"^(About|All|Metadata)")
    a = ap.parse_args()
    root = Path(a.root).expanduser()

    ours, by_pattern = catalog_sigs()
    # kind-level generative envelope: our Manchester conjunction LOWERS to one subClassOf triple
    # per conjunct — foreign per-restriction triples must be judged against conjunct KINDS.
    sub_kinds = set().union(*[_kinds_of(x) for x in ours if x.startswith("Sub[")]) if any(
        x.startswith("Sub[") for x in ours) else set()
    equiv_kinds = set().union(*[_kinds_of(x) for x in ours if x.startswith("Equiv[")]) if any(
        x.startswith("Equiv[") for x in ours) else set()
    print(f"catalog: {len(ours)} whole signatures · Sub kinds {sorted(sub_kinds)} · "
          f"Equiv kinds {sorted(equiv_kinds)}")

    def expressible(s_: str) -> bool:
        if s_ in ours:
            return True
        if s_.startswith("Sub["):
            return _kinds_of(s_) <= sub_kinds
        if s_.startswith("Equiv["):
            return _kinds_of(s_) <= equiv_kinds
        return False

    ax, examples, meta, per_module = sweep(root, a.skip)
    total = sum(ax.values())

    covered = Counter()
    blocked = Counter()
    for s_, n in ax.items():
        if expressible(s_):
            covered[s_] = n
        else:
            blocked[s_] = n
    n_cov = sum(covered.values())
    tb_total = sum(n for s_, n in ax.items() if not s_.startswith("abox:"))
    tb_cov = sum(n for s_, n in covered.items() if not s_.startswith("abox:"))

    # which of OUR pattern groups would carry the load
    pattern_load = {}
    for pat, sigs_ in by_pattern.items():
        load = sum(ax.get(s_, 0) for s_ in sigs_)
        if load:
            pattern_load[pat] = load

    out = {"root": str(root), "tag": a.tag, "n_logical_axioms": total,
           "n_covered": n_cov, "coverage": round(n_cov / max(1, total), 4),
           "tbox_rbox_total": tb_total, "tbox_rbox_covered": tb_cov,
           "tbox_rbox_coverage": round(tb_cov / max(1, tb_total), 4),
           "meta": dict(meta), "per_module": dict(per_module),
           "covered": {k: v for k, v in covered.most_common()},
           "blocked": {k: v for k, v in blocked.most_common()},
           "blocked_examples": {k: examples.get(k, []) for k, _ in blocked.most_common(30)},
           "pattern_load": pattern_load,
           "catalog_signatures": sorted(ours)}
    dest = REPO / f"build/{a.tag}_coverage.json"
    dest.write_text(json.dumps(out, indent=1))

    print(f"\n{a.tag}: {total} logical axioms · {meta.get('annotations', 0)} annotations · "
          f"{meta.get('declarations', 0)} declarations")
    print(f"COVERAGE (all logical, incl. ABox): {n_cov}/{total} = {100 * n_cov / max(1, total):.1f}%")
    print(f"COVERAGE (TBox+RBox — the schema story): {tb_cov}/{tb_total} = "
          f"{100 * tb_cov / max(1, tb_total):.1f}%\n")
    print("top covered shapes:")
    for k, v in covered.most_common(10):
        print(f"  {v:6d}  {k}")
    print("\ntop BLOCKED shapes (the not-expressible tail):")
    for k, v in blocked.most_common(18):
        ex = examples.get(k, [""])[0]
        print(f"  {v:6d}  {k:<44s} e.g. {ex[:60]}")
    print(f"\n→ {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
