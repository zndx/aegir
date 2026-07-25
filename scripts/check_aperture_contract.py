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
    # overlay (regex block form) + integration overlays (prefixed-subject TTLs,
    # rdflib-read) — the contract measures the WHOLE admission-surface vocabulary
    vocab = {**DI.load_skos(), **DI.integration_overlay_concepts()}
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
                     "code": str((p.payload or {}).get("code", "")),
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
    # integration overlays (prefixed-subject TTLs — invisible to the regex forms
    # above): the SAME S2–S9 conditions, graph-based. Layer-A external URIs are
    # inScheme members WITHOUT a skos:Concept typing in our files — §4.6.3 allows
    # this (inScheme has no such integrity condition), so S4 checks only the RANGE.
    n_int_schemes = n_int_concepts = 0
    try:
        import rdflib
        _SK = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
        gi = rdflib.Graph()
        for f in DI.integration_overlay_files():
            gi.parse(str(f))
        g_schemes = set(gi.subjects(rdflib.RDF.type, _SK.ConceptScheme))
        g_concepts = set(gi.subjects(rdflib.RDF.type, _SK.Concept))
        n_int_schemes, n_int_concepts = len(g_schemes), len(g_concepts)
        if g_schemes & g_concepts:
            s_viol.append(f"S9(integration): {len(g_schemes & g_concepts)} IRIs typed "
                          f"both Concept and ConceptScheme")
        # cross-source references are LEGAL (integration files assert memberships on
        # overlay-declared concepts and vice versa) — test against the UNION
        def _is_scheme(o):
            return o in g_schemes or str(o) in _schemes
        def _is_concept(o):
            return o in g_concepts or str(o) in _concepts
        for s, o in gi.subject_objects(_SK.hasTopConcept):
            if not _is_scheme(s):
                s_viol.append(f"S5(integration): hasTopConcept subject not a scheme: {s}")
            if not _is_concept(o):
                s_viol.append(f"S6(integration): hasTopConcept object not a concept: {o}")
        for s, o in gi.subject_objects(_SK.topConceptOf):
            if not _is_scheme(o):
                s_viol.append(f"S8(integration): topConceptOf object not a scheme: {s} → {o}")
        for s, o in gi.subject_objects(_SK.inScheme):
            if not _is_scheme(o):
                s_viol.append(f"S4(integration): inScheme object not a scheme: {s} → {o}")
        # Collection disjointness (SKOS data model: Collection ⊥ Concept, ⊥ Scheme —
        # RH 2026-07-25: the standing signal to Atlas curators that richer conceptual
        # frameworks belong in HermiT/kvasir-checked OWL, never in SKOS hierarchy claims)
        g_colls = set(gi.subjects(rdflib.RDF.type, _SK.Collection))
        for bad in g_colls & (g_concepts | g_schemes):
            s_viol.append(f"S13(integration): {bad} typed Collection AND Concept/Scheme")
        for s, o in gi.subject_objects(_SK.member):
            if not _is_concept(o) and o not in g_colls:
                s_viol.append(f"S13(integration): member not a Concept/Collection: {s} → {o}")
        # TOPIC-DOMAINS SYNC (RH 2026-07-25, instantiation-not-specialization):
        # members(sdg:TopicDomains) ≡ instances(sdg:TopicDomainConcept) — the
        # OWL ⊨ SKOS correspondence for the sector tier, plus NO broader link may
        # point at the retired apex (the false-root regression guard).
        _SDG = rdflib.Namespace("https://signals.zndx.org/sdg#")
        members = set(gi.objects(_SDG.TopicDomains, _SK.member))
        typed = set(gi.subjects(rdflib.RDF.type, _SDG.TopicDomainConcept))
        for d in members ^ typed:
            s_viol.append(f"SYNC(topic-domains): {d} "
                          f"{'member-not-typed' if d in members else 'typed-not-member'}")
        for s in gi.subjects(_SK.broader, _SDG.TopicDomain):
            s_viol.append(f"SYNC(topic-domains): {s} broader→retired apex (false root)")
        n_td = len(members & typed)
    except Exception as e:  # noqa: BLE001 — an unreadable overlay is a violation, not a crash
        s_viol.append(f"integration overlays unreadable: {e}")
        n_td = 0
    print(f"meta-structure (S2–S9+S13): schemes {len(_schemes)}+{n_int_schemes} · "
          f"concepts {len(_concepts)}+{n_int_concepts} (overlay+integration) · "
          f"topic-domains sync {n_td} members≡typed · violations {len(s_viol)}")
    for v in s_viol[:8]:
        print(f"  ✘ {v}")

    # ── recomposition drivers (RH 2026-07-23; drivers, never gates) ──
    # S9-SETTLED: the EFFECTIVE admission surface excludes scheme tops when the
    # released admission_filter arms (aperture ∩ topConcepts = ∅). GENUS-STATUS: the
    # genus-first anchor directive (skos:broader ∧ skos:narrower ∧ ¬top) measured as
    # realized — 0/N until the hierarchy deepens ("fill in as needed").
    # §4.6.3 DISCIPLINE (RH 2026-07-23): skos:hasTopConcept is CONVENTION without
    # integrity conditions — a declared topConcept need not be topmost, and atypical
    # ConceptScheme presentations (foreign/mapped/future aperture-genera schemes) may
    # declare tops that are structurally genus. Therefore: admission EXCLUSION is only
    # ever the ENUMERATED released filter set; genus status is STRUCTURAL (broader ∧
    # narrower); topConcept declarations are MEASURED against both, never trusted.
    _af = DI.armed_admission_filter()
    _excl = set(_af.get("exclude_codes") or [])
    _armed = bool(_af.get("effective_exclude"))
    _top_pts = [r for r in rows if r["top_concept"]]
    _top_codes = {r["code"] for r in _top_pts if r["code"]}
    tops_not_excluded = sorted(_top_codes - _excl)
    excluded_not_tops = sorted(_excl - _top_codes)
    _s9_settled = _armed and not tops_not_excluded
    n_genus = sum(1 for r in rows if r["broader"] and r["narrower"])
    n_pts = len(rows)
    atypical = [{"code": r["code"], "fragment": r["fragment"],
                 "top_with_broader": bool(r["broader"]),
                 "structurally_genus": bool(r["broader"] and r["narrower"])}
                for r in _top_pts if r["broader"] or (r["broader"] and r["narrower"])]
    print(f"S9-settled driver: filter {'ARMED' if _armed else 'staged (disarmed)'} · "
          f"{len(_top_pts)} declared tops · enumerated excludes {sorted(_excl)} · "
          f"declared-tops NOT in filter: {tops_not_excluded or 'none'} · "
          f"filter entries NOT declared tops: {excluded_not_tops or 'none'}")
    print(f"genus-status driver (STRUCTURAL, §4.6.3-aware): {n_genus}/{n_pts} aperture "
          f"points are genus-level (broader ∧ narrower, regardless of topConcept "
          f"declaration) · atypical top presentations: {len(atypical)}")

    (REPO / "build/aperture_contract.json").write_text(json.dumps(
        {"n": n, "broader_ok": ok_b, "narrower_ok": ok_n, "altlabel_ok": ok_a,
         "top_concept_candidates": n_top, "quality_drivers": True,
         "s9_settled": {"filter_staged": bool(_excl), "armed": _armed,
                        "declared_tops": sorted(_top_codes),
                        "enumerated_excludes": sorted(_excl),
                        "declared_tops_not_excluded": tops_not_excluded,
                        "excluded_not_declared_tops": excluded_not_tops,
                        "effective_surface_excludes_declared_tops": _s9_settled,
                        "doctrine": "exclusion is ENUMERATED (released filter), never "
                                    "derived from hasTopConcept — SKOS §4.6.3: the "
                                    "topConcept convention has no integrity conditions"},
         "genus_status": {"n_genus": n_genus, "n_points": n_pts,
                          "structural": "broader ∧ narrower, regardless of topConcept "
                                        "declaration (atypical presentations measured, "
                                        "not trusted)",
                          "atypical_top_presentations": atypical,
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
