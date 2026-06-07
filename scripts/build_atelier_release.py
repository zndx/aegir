#!/usr/bin/env python
"""Extract the Atelier-facing release from a generated corpus run.

Atelier classifies columns BLIND (by their values) into the SKOS vocabulary; we hold the
column→code reference as the scoring key. This parses each chapter's emitted JSON rows block,
pairs each table with its deterministic spine schema (the surrogate id/PK is dropped — it is
not semantic), and emits:
  - corpus_columns.parquet : RELEASE data — table / column / sample values, reference WITHHELD
  - reference.parquet      : held-back scoring key — column → true SKOS code (NOT for release)
  - release_stats.json     : scale stats for the dataset card

Deterministic. No LLM / GPU / network.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.ontology.ddl import template_to_table  # noqa: E402

JSON_RE = re.compile(r"```json\s*(.*?)```", re.S)


def load_template_map() -> dict:
    tmap = {}
    for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))):
        if "candidate" in f or "combined" in f:
            continue
        fam = Path(f).stem
        for t in load_catalog(f).templates:
            tmap[t.template_id] = (t, fam)
    return tmap


def load_code_map() -> dict:
    import pyarrow.parquet as pq
    recs = pq.read_table(REPO / "corpora/vocabulary/annotations.parquet").to_pylist()
    return {r["abbrev"].lower(): r["code"] for r in recs if r["parent_code"]}


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-run", required=True, help="dir containing chapters.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-values", type=int, default=25, help="sample values kept per column")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tmap = load_template_map()
    code_map = load_code_map()
    chapters = pq.read_table(Path(args.corpus_run) / "chapters.parquet").to_pylist()

    col_rows: list[dict] = []
    ref_rows: list[dict] = []
    n_ok = n_fail = n_tables = 0
    covered: set[str] = set()
    reasoning_tokens = 0

    for ch in chapters:
        reasoning_tokens += int(ch.get("reasoning_tokens") or 0)
        txt = ch.get("response_text") or ""
        m = JSON_RE.search(txt)
        if not m:
            n_fail += 1
            continue
        try:
            tables = json.loads(m.group(1)).get("tables", [])
        except Exception:
            n_fail += 1
            continue
        n_ok += 1
        for tb in tables:
            name = tb.get("name", "")
            rows = tb.get("rows") or []
            tid = name[2:] if name.startswith("t_") else name
            entry = tmap.get(tid)
            if not entry or not rows:
                continue
            template, fam = entry
            try:
                cols = template_to_table(template, fam).table.columns
            except Exception:
                continue
            n_tables += 1
            covered.add(tid)
            code = code_map.get(tid.lower(), "")
            for pos, c in enumerate(cols):
                if getattr(c, "slot_ref", "") == "__pk__":
                    continue  # surrogate PK — not a semantic column
                vals = [str(r[pos]) for r in rows
                        if isinstance(r, list) and len(r) > pos and r[pos] is not None]
                if not vals:
                    continue
                col_rows.append({
                    "chapter_id": ch["chapter_id"], "table_name": name,
                    "column_name": c.name, "column_position": pos,
                    "n_rows": len(vals), "sample_values": vals[:args.max_values],
                })
                ref_rows.append({
                    "chapter_id": ch["chapter_id"], "table_name": name,
                    "column_name": c.name, "reference_code": code,
                    "slot_type": getattr(c, "slot_type", ""), "template_id": tid,
                })

    pq.write_table(pa.Table.from_pylist(col_rows), out / "corpus_columns.parquet")
    pq.write_table(pa.Table.from_pylist(ref_rows), out / "reference.parquet")
    stats = {
        "n_chapters": len(chapters),
        "chapters_with_valid_json": n_ok,
        "chapters_json_fail": n_fail,
        "n_tables": n_tables,
        "n_columns": len(col_rows),
        "n_cells": sum(r["n_rows"] for r in col_rows),
        "template_coverage": len(covered),
        "template_coverage_total": 540,
        "reasoning_tokens_total": reasoning_tokens,
    }
    (out / "release_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Atelier release → {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  corpus_columns.parquet (release) + reference.parquet (HELD BACK) + release_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
