"""inc-1 — bootstrap the VALUE_GATE's value-ontology fragment from the admitted domain taxonomy.

The audit's contamination (ASHRAE-in-imaging) is a cell whose value belongs to a *different domain* than its
column. The admitted taxonomy (`build/domain_taxonomy_admitted.json`, `{admitted:[{hypernym, anchor,
members:[…]}]}`) already encodes the domain structure: each **hypernym** is an admitted domain, its **members**
are the leaf concepts that may legitimately co-occur (subtree-mixing mixes WITHIN a hypernym), and **distinct
hypernyms are the disjointness boundary** (cross-domain leakage = a violation). So the fragment is:

  classes      = members ∪ hypernyms ∪ anchors
  subclass_of  = [[member, hypernym]] + [[hypernym, anchor]]
  disjoint     = [[all hypernyms]]    # one n-ary mutual-disjointness group

This is a *strong* prior (all admitted domains mutually exclusive); a later refinement can exempt
known-compatible hypernym pairs. Members are deduped to their first hypernym so no leaf is forced under two
disjoint parents (which would make it unsatisfiable). `concept_of` maps a column's `slot_ref` → its concept.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TAXONOMY = "build/domain_taxonomy_admitted.json"


def concept_of(slot_ref: str) -> str:
    """A column's ontology concept from its ColumnSpec.slot_ref (reuses the pipeline's normalizer)."""
    from aegir.ontology.subtree_mix import concept_of as _c
    if slot_ref in ("__pk__", "", None):
        return "identifier"
    if str(slot_ref).startswith("data:"):
        return _c(str(slot_ref).rsplit(":", 1)[-1].replace("has", "", 1))
    return _c(str(slot_ref))


def value_ontology_from_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> dict:
    data = json.loads(Path(path).read_text())
    admitted = data.get("admitted", [])
    classes: set[str] = {"identifier"}
    subclass_of: list[list[str]] = []
    hypernyms: list[str] = []
    seen_member: set[str] = set()
    for entry in admitted:
        h = entry.get("hypernym")
        if not h:
            continue
        hypernyms.append(h)
        classes.add(h)
        anchor = entry.get("anchor")
        if anchor:
            classes.add(anchor)
            subclass_of.append([h, anchor])
        for m in entry.get("members", []):
            if m in seen_member or m == h:   # dedup: each leaf under exactly one hypernym
                continue
            seen_member.add(m)
            classes.add(m)
            subclass_of.append([m, h])
    hy = sorted(set(hypernyms))
    return {"classes": sorted(classes), "subclass_of": subclass_of,
            "disjoint": [hy] if len(hy) > 1 else [], "hypernyms": hy}


def broader_map_from_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> dict:
    """``{domain member → its hypernym}`` — the SKOS-broader map the subtree-mixing keys on."""
    data = json.loads(Path(path).read_text())
    return {m: e["hypernym"] for e in data.get("admitted", [])
            for m in e.get("members", []) if m != e.get("hypernym")}


def audit_mixing(entity_pools: dict, *, broader_map: dict | None = None,
                 value_onto: dict | None = None) -> dict:
    """Apply the VALUE_GATE to the subtree-mixing — the inc-1 → live link. For every mixed cell, check its
    retained SOURCE sibling against its column concept under the bootstrapped hypernym-disjointness:
    within-hypernym mixing is clean (siblings share a hypernym), a cross-hypernym source is a bad ``broader``
    edge / too-high parent and is flagged. Returns ``{n_mixed_cols, leaks:[table.col:value]}``."""
    from aegir.ontology.subtree_mix import concept_of, subtree_mixed_pools_traced
    from aegir.refine.value_gate import check_chapter
    bm = broader_map if broader_map is not None else broader_map_from_taxonomy()
    vo = value_onto if value_onto is not None else value_ontology_from_taxonomy()
    _, n_mixed, source_map = subtree_mixed_pools_traced(entity_pools, bm)
    columns = [(f"{table}.{col}", concept_of(col), list(sm.items()))
               for table, cols in source_map.items() for col, sm in cols.items()]
    kw = dict(classes=vo["classes"], subclass_of=vo["subclass_of"], disjoint=vo["disjoint"])
    leaks = check_chapter(columns, **kw)["violations"] if columns else []
    return {"n_mixed_cols": n_mixed, "leaks": leaks}


if __name__ == "__main__":
    from aegir.refine.value_gate import check_column
    onto = value_ontology_from_taxonomy()
    kw = dict(classes=onto["classes"], subclass_of=onto["subclass_of"], disjoint=onto["disjoint"])
    print(f"fragment: {len(onto['classes'])} classes, {len(onto['subclass_of'])} subclass edges, "
          f"{len(onto['hypernyms'])} mutually-disjoint hypernyms")
    # pick two leaves from two DIFFERENT hypernyms to demonstrate the bootstrapped disjointness end-to-end
    data = json.loads(Path(DEFAULT_TAXONOMY).read_text())["admitted"]
    pairs = [(e["hypernym"], e["members"][0]) for e in data if e.get("members")]
    (ha, la), (hb, lb) = pairs[0], pairs[1]
    print(f"leaf A: {la} (⊑ {ha}) | leaf B: {lb} (⊑ {hb})")
    clean = check_column(la, [("v1", la), ("v2", la)], **kw)
    cross = check_column(la, [("v1", la), ("contam", lb)], **kw)   # B-domain value in an A column
    print(f"CLEAN  column (source=concept)        : consistent={clean['consistent']}  {clean['violations']}")
    print(f"CROSS  column (B-domain value in A)   : consistent={cross['consistent']}  {cross['violations']}")
    assert clean["consistent"] and not cross["consistent"] and "contam" in cross["violations"], \
        "bootstrapped fragment failed to separate domains"
    print("\nBootstrapped value-ontology separates admitted domains (cross-domain value flagged) — inc-1 OK.")

    # audit_mixing — the VALUE_GATE applied to the subtree-mixing (cell-source-from-mixing retention).
    H1, H2 = "hyp_alpha", "hyp_beta"
    m1, m2, m3 = "leaf_a1", "leaf_a2", "leaf_b1"
    vo = {"classes": [m1, m2, m3, H1, H2], "subclass_of": [[m1, H1], [m2, H1], [m3, H2]],
          "disjoint": [[H1, H2]]}
    pools = {"t": {m1: ["va1", "va2"], m2: ["va3"], m3: ["vb1"]}}
    good = {m1: H1, m2: H1, m3: H2}                 # correct: m1,m2 ⊑ H1 ; m3 ⊑ H2
    bad = {m1: H1, m2: H1, m3: H1}                  # WRONG edge: m3 (a H2 leaf) mixed under H1
    ra, rb = audit_mixing(pools, broader_map=good, value_onto=vo), audit_mixing(pools, broader_map=bad, value_onto=vo)
    print(f"mixing audit — correct broader map      : {ra}")
    print(f"mixing audit — broken broader (m3→H1)   : {rb}")
    assert not ra["leaks"] and rb["leaks"], "audit_mixing must confirm clean / catch the cross-hypernym leak"
    pools_path = Path("src/aegir/ontology/entity_value_pools.json")
    if pools_path.exists():
        real = audit_mixing(json.loads(pools_path.read_text()))
        print(f"REAL mixing audit (entity_value_pools.json): {real['n_mixed_cols']} mixed cols, "
              f"{len(real['leaks'])} cross-hypernym leaks {real['leaks'][:5]}")
    print("audit_mixing: within-hypernym mixing CLEAN; cross-hypernym leak CAUGHT — retention link OK.")
