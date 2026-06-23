#!/usr/bin/env python
"""C3 corpus prep — serialize the REFINED relational spine to a jsonl training corpus, ENGINE-FREE.

One document per template's realized schema, with the full Path A (c) refinement stack applied
deterministically: C2 de-leaked/enriched values (entity_value_pools) + L1 domain subtree-mixing
(subtree_mix over the HermiT-admitted taxonomy) + L2 natural physical names (natural_names.json, where it
covers the base table; realized sub-tables stay semantic). No LLM/engine call → robust to run unattended.
Feeds ``tokenize_corpus_binidx`` → the C3 aug binidx; the relational signal here is exactly what the C1
domain-hypernym CTA probes (cell-values → concept), so it's the right corpus for the C3 generalization test.

    uv run --no-sync python scripts/build_refined_table_corpus.py --out /raid/build/aegir/path-a/c3/refined.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))


def _md_table(name: str, headers: "list[str]", rows: "list[list[str]]") -> str:
    head = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers)
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return f"Table: {name}\n{head}\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=0, help="cap templates/family (0=all)")
    ap.add_argument("--realize", action="store_true", default=True)
    ap.add_argument("--no-realize", dest="realize", action="store_false")
    ap.add_argument("--max-rows", type=int, default=12, help="rows serialized per table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from build_ddl_spine import catalog_files, load_spine
    from aegir.ontology import ddl as D
    from aegir.ontology.chapter_tables import definitions_for_spine, entity_pools_for_spine
    from aegir.ontology.rows import materialize_rows
    from aegir.ontology.subtree_mix import concept_of, subtree_mixed_pools

    # L1 domain broader map (member-concept → hypernym) from the HermiT-admitted taxonomy
    adm_path = REPO / "build" / "domain_taxonomy_admitted.json"
    bmap: dict[str, str] = {}
    if adm_path.exists():
        adm = json.loads(adm_path.read_text())
        for t in adm.get("admitted", []):
            for m in t["members"]:
                bmap[concept_of(m)] = t["hypernym"]

    files = catalog_files(REPO / "src/aegir/ontology/catalog")
    spine, fks, *_ = load_spine(files, a.per_family if a.per_family > 0 else None, realize=a.realize)
    pools = entity_pools_for_spine(spine)
    n_mixed = 0
    if bmap:
        pools, n_mixed = subtree_mixed_pools(pools, bmap)  # L1
    materialize_rows(spine, fks, seed=123, definitions=definitions_for_spine(spine), entity_pools=pools)  # C2

    nat_path = REPO / "src" / "aegir" / "ontology" / "natural_names.json"
    natural = json.loads(nat_path.read_text()) if nat_path.exists() else {}
    if not natural:
        print("WARN: natural_names.json absent → semantic names (run seed_natural_names --write-resource first)",
              file=sys.stderr)

    by_tid: "collections.OrderedDict[str, list]" = collections.OrderedDict()
    for st in spine:
        by_tid.setdefault(st.template.template_id, []).append(st)

    docs: list[str] = []
    for tid, sts in by_tid.items():
        nat = natural.get(tid, {})
        base_name = D.table_name(tid)
        colmap = nat.get("cols", {})
        parts = []
        for st in sts:
            rows = st.table.rows[:a.max_rows]
            if not rows:
                continue
            tname = nat.get("table", st.table.name) if (nat and st.table.name == base_name) else st.table.name
            headers = [colmap.get(c.name, c.name) for c in st.table.columns]
            parts.append(_md_table(tname, headers, rows))
        if parts:
            docs.append("\n\n".join(parts))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    chars = sum(len(d) for d in docs)
    print(f"refined corpus: {len(docs)} documents ({len(spine)} tables, {n_mixed} domain-mixed cols, "
          f"natural={'yes' if natural else 'NO'}) · ~{chars/1e6:.1f}M chars → {out}", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
