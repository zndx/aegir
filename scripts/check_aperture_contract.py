#!/usr/bin/env python
"""check_aperture_contract — the fiat SKOS contract on the admission surface (RH 2026-07-21).

Every point in the aperture qdrant collection SHALL be a skos:Concept carrying:
  1) skos:broader   — anchors are logically distinguishable, generally not top-concepts;
  2) skos:narrower  — sufficient conceptual gravity implies subordinate structure;
  3) skos:altLabel  — the chord's display form, equal to the IRI FRAGMENT (display is
     identity in the orienting instrument — the label-drift class is closed by form).

Mechanical floors, reported per anchor; --fix-mechanical authors the derivable parts into
the overlay TTL (altLabel := fragment; explicit skos:narrower from broader-inverses).
Missing broader on a top anchor is an AUTHORING decision — surfaced, never invented.

    uv run python scripts/check_aperture_contract.py [--fix-mechanical]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix-mechanical", action="store_true")
    a = ap.parse_args()
    from aegir.ontology import domain_index as DI
    vocab = DI.load_skos()
    client = DI._client(DI.DEFAULT_QDRANT_URL)
    pts, _ = client.scroll(DI.DEFAULT_APERTURE, limit=256, with_payload=True)
    narrower: "dict[str, list]" = {}
    for k, c in vocab.items():
        b = getattr(c, "broader", "") or ""
        if b:
            narrower.setdefault(b, []).append(str(k))
    rows = []
    for p in pts:
        iri = (p.payload or {}).get("iri") or ""
        frag = iri.rsplit("#", 1)[-1]
        c = vocab.get(iri)
        has_broader = bool(getattr(c, "broader", "")) if c else False
        is_top = bool(getattr(c, "top_concept_of", "")) if c else False
        has_narrower = bool(narrower.get(iri))
        alt = (getattr(c, "alt_label", "") or "") if c else ""
        alt_ok = alt == frag
        rows.append({"point": p.id, "iri": iri, "fragment": frag,
                     "in_vocab": c is not None, "broader": has_broader,
                     "top_concept": is_top,
                     "hierarchy_position": has_broader or is_top,
                     "narrower": has_narrower, "altLabel_eq_fragment": alt_ok,
                     "altLabel": alt})
    n = len(rows)
    ok_b = sum(r["broader"] for r in rows)
    n_top = sum(r["top_concept"] for r in rows)
    ok_n = sum(r["narrower"] for r in rows)
    ok_a = sum(r["altLabel_eq_fragment"] for r in rows)
    print(f"aperture contract (QUALITY DRIVERS, not gates — RH 2026-07-21): {n} points · "
          f"broader {ok_b}/{n} · narrower {ok_n}/{n} · altLabel≡fragment {ok_a}/{n} · "
          f"top-concept CANDIDATES {n_top} (pending rdfs:domain → ConceptScheme confirmation; "
          f"S9 disjointness may retire them from the aperture)")
    for r in rows:
        flags = [k for k in ("broader", "narrower", "altLabel_eq_fragment") if not r[k]]
        if flags or not r["in_vocab"]:
            print(f"  ✘ {r['fragment'][:44]:<44} missing: "
                  f"{', '.join(flags) or 'NOT IN VOCAB'}")
    # ── META-LEVEL well-formedness (S2–S9; RH 2026-07-21): the SKOS vocabulary's OWN
    # axiomatization over our scheme expressions — distinct from the OBJECT-level rdfs:domain
    # we create on sdg properties. hasTopConcept: domain ConceptScheme (S5), range Concept
    # (S6); topConceptOf inverse (S7/S8); inScheme range ConceptScheme (S4); S9 disjointness.
    import re as _re
    t_all = "\n".join(Path(f).read_text() for f in
                       (DI.DEFAULT_OVERLAY, REPO / "corpora/vocabulary/vocabulary.ttl")
                       if Path(f).exists())
    _schemes = set(_re.findall(r"<([^>]+)> a skos:ConceptScheme", t_all))
    _concepts = set(_re.findall(r"<([^>]+)> a skos:Concept\b", t_all))
    s_viol = []
    if _schemes & _concepts:
        s_viol.append(f"S9: {len(_schemes & _concepts)} IRIs typed both Concept and ConceptScheme")
    for m in _re.finditer(r"<([^>]+)>[^.]*?skos:hasTopConcept((?:\s*<[^>]+>\s*,?)+)", t_all, _re.S):
        if m.group(1) not in _schemes:
            s_viol.append(f"S5: hasTopConcept subject not a scheme: {m.group(1)}")
        s_viol += [f"S6: hasTopConcept object not a concept: {o}"
                   for o in _re.findall(r"<([^>]+)>", m.group(2)) if o not in _concepts]
    for m in _re.finditer(r"<([^>]+)> a skos:Concept\b(.*?)(?=\n<|\Z)", t_all, _re.S):
        s_viol += [f"S8: topConceptOf object not a scheme: {m.group(1)} → {o}"
                   for o in _re.findall(r"skos:topConceptOf <([^>]+)>", m.group(2))
                   if o not in _schemes]
        s_viol += [f"S4: inScheme object not a scheme: {m.group(1)} → {o}"
                   for o in _re.findall(r"skos:inScheme <([^>]+)>", m.group(2))
                   if o not in _schemes]
    print(f"meta-structure (S2–S9): schemes {len(_schemes)} · concepts {len(_concepts)} · "
          f"violations {len(s_viol)}")
    for v in s_viol[:8]:
        print(f"  ✘ {v}")

    # ── recomposition drivers (RH 2026-07-23; drivers, never gates) ──
    # S9-SETTLED: the EFFECTIVE admission surface excludes scheme tops when the
    # released admission_filter arms (aperture ∩ topConcepts = ∅). GENUS-STATUS: the
    # genus-first anchor directive (skos:broader ∧ skos:narrower ∧ ¬top) measured as
    # realized — 0/N until the hierarchy deepens ("fill in as needed").
    _af = DI.armed_admission_filter()
    _excl = set(_af.get("exclude_codes") or [])
    _armed = bool(_af.get("effective_exclude"))
    _top_pts = [r for r in rows if r["top_concept"]]
    _s9_settled = _armed or not _top_pts
    n_genus = sum(1 for r in rows if r["broader"] and r["narrower"]
                  and not r["top_concept"])
    n_nonroot = sum(1 for r in rows if not r["top_concept"])
    print(f"S9-settled driver: filter {'ARMED' if _armed else 'staged (disarmed)'} · "
          f"{len(_top_pts)} top-concept points in the collection · effective admission "
          f"surface {'excludes tops ✓' if _s9_settled else 'still includes tops'}")
    print(f"genus-status driver: {n_genus}/{n_nonroot} non-root anchors are genus-level "
          "(broader ∧ narrower) — deepening converges this toward full-depth support")

    (REPO / "build/aperture_contract.json").write_text(json.dumps(
        {"n": n, "broader_ok": ok_b, "narrower_ok": ok_n, "altlabel_ok": ok_a,
         "top_concept_candidates": n_top, "quality_drivers": True,
         "s9_settled": {"filter_staged": bool(_excl), "armed": _armed,
                        "top_points_in_collection": len(_top_pts),
                        "effective_surface_excludes_tops": _s9_settled},
         "genus_status": {"n_genus": n_genus, "n_nonroot": n_nonroot,
                          "directive": "anchors gain genus status as skos:narrower "
                                       "children are authored (fill in as needed)"},
         "meta_s2_s9": {"schemes": len(_schemes), "concepts": len(_concepts),
                        "violations": s_viol}, "rows": rows}, indent=1))
    print("→ build/aperture_contract.json")

    if a.fix_mechanical:
        ttl_p = DI.DEFAULT_OVERLAY
        t = ttl_p.read_text()
        fixed = 0
        for r in rows:
            if not r["in_vocab"]:
                continue
            blk_key = f"<{r['iri']}> a skos:Concept"
            if blk_key not in t:
                continue
            if not r["altLabel_eq_fragment"]:
                t = t.replace(blk_key + " ;",
                              blk_key + f' ;\n    skos:altLabel "{r["fragment"]}" ;', 1)
                fixed += 1
        # explicit narrower inverses for anchors that have children but no explicit property
        for r in rows:
            kids = narrower.get(r["iri"], [])
            if kids and f"<{r['iri']}> a skos:Concept" in t and "skos:narrower" not in \
                    t.split(f"<{r['iri']}> a skos:Concept", 1)[1][:600]:
                nar = " ,\n        ".join(f"<{k}>" for k in kids[:12])
                t = t.replace(f"<{r['iri']}> a skos:Concept ;",
                              f"<{r['iri']}> a skos:Concept ;\n    skos:narrower {nar} ;", 1)
                fixed += 1
        ttl_p.write_text(t)
        print(f"--fix-mechanical: {fixed} overlay edits (altLabel:=fragment; explicit narrower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
