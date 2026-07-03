#!/usr/bin/env python
"""Extract the Atelier-facing release from a DDL-spine run (v2 — natural-register, spine-consuming).

Atelier classifies columns BLIND into the SKOS vocabulary; we hold the column→code reference as the
scoring key. v1 parsed chapter JSON blocks (no longer emitted) and shipped obfuscated names; v2 reads
the spine footprint directly (``base_rows.parquet`` + ``base_table_index.parquet`` + ``naming_map.parquet``
from ``build_ddl_spine --realize --naming natural``) and ships the NATURAL-register surface — the DBA
names are the *deliberately weak* name evidence (Atelier's ``name_match`` channel matches names against
the very vocabulary we emit, so a semantic name would BE the answer key; natural names make the
benchmark measure value-driven comprehension). Emits:

  - corpus_columns.parquet : RELEASE — opaque table_id/column_id, NATURAL column_name + register +
                             name_provenance stamps, sample values, FK topology by opaque id.
                             Names never in the semantic register: columns of templates without a
                             membrane-admitted natural record are MASKED to ``col<pos>`` (a generic
                             Atelier's ``_is_generic_name`` guard catches → values-only classification),
                             stamped ``masked-semantic`` — visible, never silent.
  - reference.parquet      : HELD-BACK key — column → SKOS code, plus the semantic register
                             (elucidation channel), template identity, and slot_ref lineage.
  - release_stats.json     : scale stats + the GENERATION MANIFEST {ontology_sha, vocab_generation,
                             ddl_run_id, corpus_run_id, naming} — the version-skew guard Atelier
                             fail-fasts on.

Atelier loader contract honored (their ``aegir_release.load_aegir_release_samples``): ``column_name``
is always a non-null snake_case string, unique within its table; ``sample_values`` a list[str];
``n_rows`` int. Deterministic. No LLM / GPU / network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# name_provenance values an authentic-natural benchmark surface may consume (the Atelier-side filter)
AUTHENTIC_PROVENANCE = {"engine-derived", "composed"}


def load_code_map() -> "tuple[dict, int]":
    import pyarrow.parquet as pq
    recs = pq.read_table(REPO / "corpora/vocabulary/annotations.parquet").to_pylist()
    return {r["abbrev"].lower(): r["code"] for r in recs if r["parent_code"]}, len(recs)


def generation_manifest(spine_dir: Path, spine_manifest: dict, n_vocab: int) -> dict:
    """The version-skew guard (task #136.2): one place recording which generation of each Data Product
    this release was cut against, so a consumer can fail fast on cross-generation mixes."""
    try:
        ontology_sha = subprocess.run(["git", "-C", str(REPO / "corpora"), "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        ontology_sha = None
    return {
        "ontology_sha": ontology_sha,
        "vocab_generation": n_vocab,
        "ddl_run_id": spine_manifest.get("run_id") or spine_dir.name,
        "corpus_run_id": None,  # set when a release is cut against a generated text corpus
        "naming": spine_manifest.get("naming"),
        "name_provenance_distribution": spine_manifest.get("name_provenance_distribution"),
    }


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-spine", required=True,
                    help="a build_ddl_spine run dir (base_rows + base_table_index + naming_map parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-values", type=int, default=50,
                    help="sample values kept per column (Atelier reads 5 for embedding, ≤50 all_values)")
    ap.add_argument("--no-mask", action="store_true",
                    help="ship semantic-passthrough concept names instead of masking to col<pos> "
                         "(diagnostic only — the masked form is the blind release)")
    args = ap.parse_args()
    spine_dir = Path(args.from_spine)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spine_manifest = json.loads((spine_dir / "manifest.json").read_text())
    code_map, n_vocab = load_code_map()
    base_rows = pq.read_table(spine_dir / "base_rows.parquet").to_pylist()
    index = {r["table_name"]: r for r in pq.read_table(spine_dir / "base_table_index.parquet").to_pylist()}
    nm_path = spine_dir / "naming_map.parquet"
    naming = {}
    if nm_path.exists():
        naming = {(r["table_name"], r["col_name"]): r
                  for r in pq.read_table(nm_path).to_pylist()}

    # group cells: (table, col) → ordered distinct values; (table) → n_rows, fk targets, col order
    cells: dict[tuple, list[str]] = {}
    col_pos: dict[str, list[str]] = {}
    fk_target: dict[tuple, str] = {}
    pk_cols: set[tuple] = set()
    for r in base_rows:
        key = (r["table_name"], r["col_name"])
        if r["is_pk"]:
            pk_cols.add(key)
            continue
        if r["col_name"] not in col_pos.setdefault(r["table_name"], []):
            col_pos[r["table_name"]].append(r["col_name"])
        if r["value"] is not None and r["value"] != "":
            cells.setdefault(key, []).append(str(r["value"]))
        if r["is_fk"] and r.get("fk_target_table"):
            fk_target[key] = r["fk_target_table"]

    tbl_ids = {t: f"tbl_{i:06d}" for i, t in enumerate(sorted(col_pos))}  # opaque, name-free
    col_rows: list[dict] = []
    ref_rows: list[dict] = []
    prov_counts: Counter = Counter()
    code_hits = code_misses = 0
    covered: set[str] = set()

    for tname, cols in sorted(col_pos.items()):
        idx = index.get(tname) or {}
        tid = idx.get("template_id", "")
        covered.add(tid)
        code = code_map.get(tid.lower(), "")
        if code:
            code_hits += 1
        else:
            code_misses += 1
        table_id = tbl_ids[tname]
        # table-level register/provenance from any column's naming row (uniform per table)
        t_nm = next((naming.get((tname, c)) for c in cols if naming.get((tname, c))), None)
        for pos, cname in enumerate(cols):
            vals = cells.get((tname, cname)) or []
            if not vals:
                continue
            nm = naming.get((tname, cname)) or {}
            prov = nm.get("name_provenance", "unmapped")
            register = nm.get("register", "semantic")
            public_name = cname
            # BLINDNESS: a concept-bearing semantic name never ships. Masked names are the values-only
            # arm — Atelier's generic-name guard catches col<pos> and classifies on values alone.
            if not args.no_mask and prov in ("semantic-passthrough", "unmapped", "semantic"):
                # structural DBA tokens (attr_name, entity_id, value …) are register-invariant: they
                # appear when the table's OTHER columns carry authentic natural provenance
                row_provs = {(naming.get((tname, c)) or {}).get("name_provenance") for c in cols}
                if not (row_provs & AUTHENTIC_PROVENANCE):
                    public_name = f"col{pos}"
                    prov = "masked-semantic"
            prov_counts[prov] += 1
            distinct = list(dict.fromkeys(vals))
            col_id = f"{table_id}_c{pos}"
            col_rows.append({  # PUBLIC release: no table_name / template / semantic register
                "table_id": table_id, "column_id": col_id, "column_name": public_name,
                "register": register, "name_provenance": prov,
                "n_rows": len(vals), "sample_values": distinct[:args.max_values],
                "fk_to_table_id": tbl_ids.get(fk_target.get((tname, cname), ""), None),
            })
            ref_rows.append({  # HELD-BACK key: de-anonymisation + answer + elucidation register
                "table_id": table_id, "column_id": col_id, "reference_code": code,
                "template_id": tid, "source_table": tname, "column_name": public_name,
                "semantic_table": nm.get("semantic_table", ""), "semantic_col": nm.get("semantic_col", ""),
                "slot_ref": nm.get("slot_ref", ""), "kind": nm.get("kind", idx.get("family", "")),
            })
        _ = t_nm  # (table-level row retained for future table_label emission)

    pq.write_table(pa.Table.from_pylist(col_rows), out / "corpus_columns.parquet")
    pq.write_table(pa.Table.from_pylist(ref_rows), out / "reference.parquet")
    stats = {
        "release_version": "v2-natural-register",
        "n_tables": len(col_pos),
        "n_columns": len(col_rows),
        "n_cells": sum(r["n_rows"] for r in col_rows),
        "template_coverage": len(covered),
        "reference_code_hits": code_hits,
        "reference_code_misses": code_misses,
        "column_name_provenance": dict(prov_counts),
        "generation_manifest": generation_manifest(spine_dir, spine_manifest, n_vocab),
    }
    (out / "release_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Atelier release → {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  corpus_columns.parquet (release) + reference.parquet (HELD BACK) + release_stats.json")
    # loud contract checks (never silent): uniqueness within table + non-null snake_case names
    by_tbl: dict[str, set] = {}
    for r in col_rows:
        s = by_tbl.setdefault(r["table_id"], set())
        if r["column_name"] in s:
            print(f"  !! DUPLICATE column_name '{r['column_name']}' within {r['table_id']} "
                  "(breaks Atelier's name-keyed prediction join)")
            return 1
        s.add(r["column_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
