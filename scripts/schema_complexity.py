#!/usr/bin/env python
"""Quantify the STRUCTURAL complexity of the corpus's relational schema — columns per table,
foreign-key density, and the schema-graph shape — so we can compare to real-world ("in the
wild") distributions and drive generation toward common patterns.

Deterministic (no LLM / GPU). Measures:

  Per table (template → table):
    - column count (width); data-property vs class-reference columns; ObjectProperty relations.
  FK graph (cross-family edges from the spine + intra-template ObjectProperty FKs):
    - FKs per table (out-degree), referenced-by (in-degree, = hubs),
    - weakly-connected components (schema fragmentation), longest FK chain (normalization depth).
  Per chapter (optional --corpus-run):
    - tables/chapter and whether a chapter's cited tables form a connected FK sub-schema
      (schema coherence per document).

Wild reference (rules of thumb from SchemaPile / real RDBMS corpora): real tables run ~5-30
columns (median ~7-10); real schemas show star/snowflake hubs, normalized FK chains, and
junction (M:N) tables. Pass --schemapile <parquet> to compute the empirical target instead.
"""
from __future__ import annotations

import argparse
import glob
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.complex import FamilyComplex  # noqa: E402
from aegir.ontology.ddl import template_to_table, cross_family_fks, parse_restrictions  # noqa: E402


def load_catalog_objs():
    objs, fam_of = {}, {}
    for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))):
        if "candidate" in f or "combined" in f:
            continue
        fam = Path(f).stem
        cat = load_catalog(f)
        for t in cat.templates:
            objs[t.template_id] = t
            fam_of[t.template_id] = fam
    return objs, fam_of


def components(nodes: set, edges: list[tuple]) -> list[set]:
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comp = defaultdict(set)
    for n in nodes:
        comp[find(n)].add(n)
    return list(comp.values())


def longest_chain(nodes: set, edges: list[tuple]) -> int:
    """Longest directed FK path (DAG-ish; cycle-safe via visited depth memo)."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    memo, onstack = {}, set()
    def depth(n):
        if n in memo:
            return memo[n]
        if n in onstack:
            return 0  # cycle guard
        onstack.add(n)
        d = 1 + max((depth(m) for m in adj[n]), default=0)
        onstack.discard(n); memo[n] = d
        return d
    return max((depth(n) for n in nodes), default=0)


def summarize(label, vals):
    if not vals:
        print(f"  {label:<30} (none)"); return
    vals = sorted(vals)
    p = lambda q: vals[min(len(vals) - 1, int(q * len(vals)))]
    print(f"  {label:<30} mean {st.mean(vals):5.2f} | median {st.median(vals):4.0f} | "
          f"p90 {p(0.9):4.0f} | max {max(vals):4.0f}")


def measure_schemapile(path: str, sample: int = 3000) -> None:
    """Empirical real-world target distribution from the SchemaPile structured parquet."""
    import random
    import pyarrow.parquet as pq
    rows = pq.read_table(path, columns=["TABLES"]).to_pylist()
    rng = random.Random(0)
    if len(rows) > sample:
        rows = rng.sample(rows, sample)
    cpt, tps, types = [], [], Counter()
    fk_tables = ntot = 0
    for r in rows:
        tabs = r.get("TABLES") or []
        tps.append(len(tabs))
        for tb in tabs:
            cols = tb.get("COLUMNS") or []
            cpt.append(len(cols))
            ntot += 1
            if tb.get("FOREIGN_KEYS"):
                fk_tables += 1
            for c in cols:
                types[str(c.get("TYPE"))] += 1
    print(f"  (SchemaPile, {len(rows)} real schemas)")
    summarize("columns/table (real)", cpt)
    summarize("tables/schema (real)", tps)
    print(f"  top column types: {[t for t, _ in types.most_common(10)]}")
    print(f"  tables with a FOREIGN_KEYS list: {100 * fk_tables / max(ntot, 1):.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-run", nargs="*", default=[],
                    help="dir(s) with chapters.parquet for per-chapter coherence")
    ap.add_argument("--realized-only", action="store_true",
                    help="restrict to templates actually realized in --corpus-run")
    ap.add_argument("--schemapile", default=None,
                    help="SchemaPile structured parquet — measure the real target distribution")
    args = ap.parse_args()

    objs, fam_of = load_catalog_objs()
    fc = FamilyComplex.load_optional(str(REPO / "src/aegir/ontology/family_complex.json"))

    # realized set from the corpus (if given)
    realized = set()
    chapter_templates = []
    for run in args.corpus_run:
        import pyarrow.parquet as pq
        for r in pq.read_table(Path(run) / "chapters.parquet").to_pylist():
            tids = list(r.get("template_ids") or [])
            chapter_templates.append(tids)
            realized.update(tids)
    universe = (realized & set(objs)) if (args.realized_only and realized) else set(objs)

    # per-table width + FK out (ObjectProperty restrictions) ----------------------
    width, data_cols, ref_cols, objprops = [], [], [], []
    spines = []
    tbl_name = {}
    for tid in universe:
        t = objs[tid]
        spine = template_to_table(t, fam_of[tid])
        spines.append(spine)
        cols = spine.table.columns
        tbl_name[spine.table.name] = tid
        width.append(len(cols))
        data_cols.append(sum(1 for c in cols if getattr(c, "slot_type", "").startswith("xsd:")
                             or getattr(c, "slot_type", "") == "DataProperty"))
        ref_cols.append(sum(1 for c in cols if getattr(c, "slot_type", "") in ("Class", "Individual")))
        objprops.append(sum(1 for r in parse_restrictions(t) if r.is_slot))

    # FK graph (cross-family, family-complex gated) -------------------------------
    edges, _ = cross_family_fks(spines, fc)
    names = set(tbl_name)
    fk_edges = [(e.src_table, e.dst_table) for e in edges
                if e.src_table in names and e.dst_table in names]
    out_deg = Counter(); in_deg = Counter()
    for a, b in fk_edges:
        out_deg[a] += 1; in_deg[b] += 1
    comps = components(names, fk_edges)
    chain = longest_chain(names, fk_edges)

    n = len(universe)
    print(f"SCHEMA COMPLEXITY  ({n} table-types{' realized in corpus' if args.realized_only else ' in spine'})\n")
    print("TABLE WIDTH")
    summarize("columns / table (incl id)", width)
    summarize("data-property columns", data_cols)
    print(f"  {'reference (FK-ish) columns':<30} mean {st.mean(ref_cols):.2f}")
    print("FOREIGN-KEY GRAPH")
    print(f"  total FK edges: {len(fk_edges)}  ({len(fk_edges)/n:.2f} per table)")
    summarize("FKs out / table", [out_deg[x] for x in names])
    print(f"  hub tables (in-degree ≥3): {sum(1 for x in names if in_deg[x] >= 3)} | "
          f"max in-degree {max(in_deg.values(), default=0)}")
    print(f"  weakly-connected components: {len(comps)}  "
          f"(largest {max((len(c) for c in comps), default=0)}, singletons {sum(1 for c in comps if len(c)==1)})")
    print(f"  longest FK chain (depth): {chain}")
    print(f"  M:N ObjectProperty relations (junction-table candidates): {sum(objprops)}")

    # per-chapter coherence -------------------------------------------------------
    if chapter_templates:
        tpl_set = set(tbl_name.values())
        # build template-level FK adjacency
        tid_edges = set()
        for a, b in fk_edges:
            tid_edges.add((tbl_name[a], tbl_name[b]))
        per_n, connected = [], 0
        for tids in chapter_templates:
            ts = [t for t in tids if t in tpl_set]
            per_n.append(len(ts))
            if len(ts) >= 2:
                sub = [(a, b) for (a, b) in tid_edges if a in ts and b in ts]
                cc = components(set(ts), sub)
                if len(cc) == 1:
                    connected += 1
        print("\nPER-CHAPTER SCHEMA")
        summarize("tables / chapter", per_n)
        multi = sum(1 for x in per_n if x >= 2)
        print(f"  chapters whose tables form ONE connected FK schema: "
              f"{connected}/{multi} ({100*connected/max(multi,1):.0f}% of multi-table chapters)")

    print("\nWILD REFERENCE (real-world target):")
    if args.schemapile:
        measure_schemapile(args.schemapile)
    else:
        print("  SchemaPile rules of thumb — columns/table median ~5-7 (we are narrow); real schemas")
        print("  show typed columns (Varchar>Int>Timestamp>Text>Date), FK-connected tables, junction tables.")
        print("  (pass --schemapile <parquet> for the measured distribution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
