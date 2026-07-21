#!/usr/bin/env python
"""check_triad_entailment — the SHACL leg: realize OWL ⊨ SKOS ⊨ SHACL (RH 2026-07-21; #16).

The triad's coherence, measured mechanically and DRIVER-FRAMED (quality ratios surfaced,
regressions gateable later once floors settle):

  OWL ⊨ SKOS   — every non-deprecated generated SKOS concept's defining template head exists
                 as a class in the realized ontology (the vocab is a projection of the OWL,
                 never independent vocabulary).
  OWL ⊨ SHACL  — every sh:targetClass names a realized OWL class; object-property shape
                 constraints (sh:class / sh:minCount / sh:maxCount) correspond to the class's
                 realized restrictions (some → minCount≥1; exactly n → min=max=n; filler ⊑
                 sh:class); mismatches enumerated by kind.
  SKOS ⊨ SHACL — sh:in enumerations correspond to the SKOS layer: value tokens matched
                 against concept labels/altLabels (the coded-concept binding made visible).

    uv run python scripts/check_triad_entailment.py [--shapes <shapes.ttl>] [--json out]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_SHAPES = "/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06/ontology/shapes.ttl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owl", default="",
                    help="realized owl; default = the shapes' OWN run dir (same-generation "
                         "pairing — shapes entail against the ontology they were projected from)")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--json", default=str(REPO / "build/triad_entailment.json"))
    a = ap.parse_args()

    from rdflib import Graph, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, SH

    # same-generation pairing: the run emits OMN (machine-regular) — read classes +
    # restrictions textually; an .owl neighbor is preferred when present
    run_dir = Path(a.shapes).parent
    owl_path = a.owl or (str(run_dir / "sdg-ontology.owl")
                         if (run_dir / "sdg-ontology.owl").exists()
                         else str(run_dir / "sdg-ontology.omn"))
    classes: set = set()
    rest: "dict[str, dict]" = {}
    if owl_path.endswith(".omn"):
        txt = Path(owl_path).read_text()
        pfx = dict(re.findall(r"Prefix:\s*(\w+):\s*<([^>]+)>", txt))

        def _x(curie: str) -> str:
            if curie.startswith("<"):
                return curie.strip("<>")
            k, _, l_ = curie.partition(":")
            return pfx.get(k, k + ":") + l_

        for m in re.finditer(r"^Class:\s*(\S+)\n((?:    .*\n)*)", txt, re.M):
            ciri = _x(m.group(1))
            classes.add(ciri)
            body = m.group(2)
            for rm in re.finditer(
                    r"(\S+)\s+(some|only|exactly|min|max)\s*(\d+)?\s+(\S+?)(?:,|\s|$)", body):
                prop, quant, num, filler = rm.groups()
                if prop.startswith(("Annotations", "SubClassOf", "EquivalentTo")):
                    continue
                ent = rest.setdefault(ciri, {}).setdefault(_x(prop), {
                    "kinds": set(), "fillers": set(), "min": None, "max": None})
                ent["kinds"].add(quant)
                f_ = _x(filler.rstrip(","))
                if not f_.startswith("http://www.w3.org/2001/XMLSchema#"):
                    ent["fillers"].add(f_)
                if quant == "exactly" and num:
                    ent["min"] = ent["max"] = int(num)
                elif quant == "min" and num:
                    ent["min"] = int(num)
                elif quant == "max" and num:
                    ent["max"] = int(num)
    else:
        g = Graph()
        g.parse(owl_path)
        classes = {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}

    # realized restriction index (rdflib branch): class → prop → constraints
    def _first(s_, p_):
        for o in g.objects(s_, p_):
            return o
        return None

    for s_ in (g.subjects(RDF.type, OWL.Class) if not owl_path.endswith(".omn") else []):
        if not isinstance(s_, URIRef):
            continue
        for o in list(g.objects(s_, RDFS.subClassOf)) + list(g.objects(s_, OWL.equivalentClass)):
            todo = [o]
            while todo:
                e = todo.pop()
                if (e, OWL.intersectionOf, None) in g:
                    lst = _first(e, OWL.intersectionOf)
                    while lst and lst != RDF.nil:
                        todo.append(_first(lst, RDF.first))
                        lst = _first(lst, RDF.rest)
                    continue
                p_ = _first(e, OWL.onProperty)
                if p_ is None:
                    continue
                ent = rest.setdefault(str(s_), {}).setdefault(str(p_), {"kinds": set(), "fillers": set(),
                                                                        "min": None, "max": None})
                if _first(e, OWL.someValuesFrom) is not None:
                    ent["kinds"].add("some")
                    f = _first(e, OWL.someValuesFrom)
                    if isinstance(f, URIRef):
                        ent["fillers"].add(str(f))
                for pred, key in ((OWL.qualifiedCardinality, "exact"),
                                  (OWL.minQualifiedCardinality, "min"),
                                  (OWL.maxQualifiedCardinality, "max")):
                    n_ = _first(e, pred)
                    if n_ is not None:
                        ent["kinds"].add(key)
                        v = int(n_)
                        if key in ("exact", "min"):
                            ent["min"] = v
                        if key in ("exact", "max"):
                            ent["max"] = v
                        oc = _first(e, OWL.onClass)
                        if isinstance(oc, URIRef):
                            ent["fillers"].add(str(oc))

    # ── OWL ⊨ SKOS ────────────────────────────────────────────────────────────
    from aegir.ontology import domain_index as DI
    from aegir.ontology.schema import load_catalog
    cat = load_catalog(REPO / "src/aegir/ontology/catalog/catalog.json")
    head_of = {}
    for t in cat.templates:
        m = re.search(r"Class:\s*(?:\{(\w+):Class\}|sdg:(\w+))", t.manchester_template or "")
        if m:
            head_of[t.template_id.upper()] = m.group(1) or m.group(2)
    vocab = DI.load_skos()
    # the SKOS leg pairs with the CATALOG realization (vocab is ITS projection); the run
    # realization carries the shapes. The GENERATION SPLIT is itself a measured driver.
    catg = Graph()
    catg.parse(str(REPO / "corpora/ontology/sdg-ontology.owl"))
    cat_classes = {str(x) for x in catg.subjects(RDF.type, OWL.Class) if isinstance(x, URIRef)}
    sk_total = sk_ok = 0
    sk_miss = []
    for k, c in vocab.items():
        if getattr(c, "deprecated", False):
            continue
        loc = str(k).rsplit("#", 1)[-1]
        tid_u = next((suf for i, ch in enumerate(loc) if ch == "_"
                      and (suf := loc[i + 1:]) in head_of), None)
        if tid_u is None:
            continue                                    # upper/domain concepts — no template
        sk_total += 1
        head = head_of[tid_u]
        if f"https://signals.zndx.org/sdg#{head}" in cat_classes:
            sk_ok += 1
        else:
            sk_miss.append(loc)

    # ── OWL ⊨ SHACL + SKOS ⊨ SHACL ────────────────────────────────────────────
    sg = Graph()
    sg.parse(a.shapes)
    n_shapes = tc_ok = 0
    tc_miss = []
    corr = Counter()
    mism = Counter()
    mism_ex: "dict[str, list]" = {}
    labels_all: set = set()
    for c in vocab.values():
        for lab in ([getattr(c, "pref_label", "")] +
                    ([getattr(c, "alt_label", "")] if getattr(c, "alt_label", "") else [])):
            if lab:
                labels_all.add(lab.strip().lower())
    in_total = in_hit = 0
    for sh_node in sg.subjects(RDF.type, SH.NodeShape):
        n_shapes += 1
        tc = next(iter(sg.objects(sh_node, SH.targetClass)), None)
        if tc is None:
            continue
        if str(tc) in classes:
            tc_ok += 1
        else:
            tc_miss.append(str(tc).rsplit("#", 1)[-1])
        crest = rest.get(str(tc), {})
        for prop in sg.objects(sh_node, SH.property):
            path = next(iter(sg.objects(prop, SH.path)), None)
            sh_cls = next(iter(sg.objects(prop, URIRef(str(SH) + "class"))), None)
            mn = next(iter(sg.objects(prop, SH.minCount)), None)
            mx = next(iter(sg.objects(prop, SH.maxCount)), None)
            inlist = next(iter(sg.objects(prop, URIRef(str(SH) + "in"))), None)
            if inlist is not None:
                lst = inlist
                while lst and lst != RDF.nil:
                    v = next(iter(sg.objects(lst, RDF.first)), None)
                    if v is not None:
                        in_total += 1
                        if str(v).strip().lower() in labels_all:
                            in_hit += 1
                    lst = next(iter(sg.objects(lst, RDF.rest)), None)
            if sh_cls is None and mn is None and mx is None:
                continue                                # datatype-only property shape
            ent = crest.get(str(path)) if path is not None else None
            if ent is None:
                mism["shape_constraint_without_owl_restriction"] += 1
                mism_ex.setdefault("shape_constraint_without_owl_restriction", []).append(
                    f"{str(tc).rsplit('#', 1)[-1]}.{str(path).rsplit('#', 1)[-1]}")
                continue
            ok = True
            if sh_cls is not None and str(sh_cls) not in ent["fillers"]:
                ok = False
                mism["filler_mismatch"] += 1
                mism_ex.setdefault("filler_mismatch", []).append(
                    f"{str(tc).rsplit('#', 1)[-1]}.{str(path).rsplit('#', 1)[-1]}")
            if mn is not None and ent["min"] is not None and int(mn) != ent["min"]:
                ok = False
                mism["mincount_mismatch"] += 1
            if mx is not None and ent["max"] is not None and int(mx) != ent["max"]:
                ok = False
                mism["maxcount_mismatch"] += 1
            if ok:
                corr["constraint_corresponds"] += 1

    overlap = len(cat_classes & classes)
    out = {"run_classes": len(classes), "catalog_classes": len(cat_classes),
           "generation_overlap": overlap,
           "owl_skos": {"projected": sk_total, "entailed": sk_ok,
                        "missing": sk_miss[:12]},
           "owl_shacl": {"shapes": n_shapes, "targetclass_realized": tc_ok,
                         "targetclass_missing": tc_miss[:12],
                         "constraints_corresponding": corr.get("constraint_corresponds", 0),
                         "mismatches": dict(mism),
                         "mismatch_examples": {k: v[:6] for k, v in mism_ex.items()}},
           "skos_shacl": {"sh_in_values": in_total, "matched_to_skos_labels": in_hit},
           "driver_framed": True}
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"TRIAD (drivers): catalog OWL {len(cat_classes)} · run OWL {len(classes)} · "
          f"GENERATION OVERLAP {overlap} — the unification gap is the composition target")
    print(f"  OWL⊨SKOS   {sk_ok}/{sk_total} (vs the catalog realization — the vocab's source)"
          + (f" · missing e.g. {sk_miss[:3]}" if sk_miss else ""))
    print(f"  OWL⊨SHACL  targetClass {tc_ok}/{n_shapes} · constraints corresponding "
          f"{corr.get('constraint_corresponds', 0)} · mismatches {dict(mism) or 'none'}")
    print(f"  SKOS⊨SHACL sh:in∩SKOS-labels {in_hit}/{in_total} (sh:in draws from VALUE "
          "POOLS; the SKOS overlap is the coded-concept binding subset, not a target of 100%)")
    print(f"→ {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
