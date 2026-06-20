#!/usr/bin/env python
"""Build the ontology → SQL DDL spine (the syntactic axis of the corpus).

Deterministically lowers every catalog template to a relational table, renders
real SQL DDL (a cross-dialect Trino∩Spark statement + a Spark Iceberg-flavored
variant), validates each with ``polyglot`` (the syntactic gate, analogous to the
ontology's R_axiom), and inventories the SQL features exercised (the syntactic
coverage axis, analogous to ``ontology_coverage_audit.py``). Cross-family foreign
keys — the join structure the corpus views exploit — are sanctioned by the
empirical family complex.

Outputs under ``--output-dir/<run_id>/`` (Iceberg-ready parquet + manifest):
  - ddl_statements.parquet      — one row per table (canonical + iceberg DDL)
  - ddl_validation.parquet      — one row per (table × dialect × variant)
  - sql_feature_coverage.parquet — one row per (feature × variant)
  - cross_family_fks.parquet    — emitted + suppressed FK candidates (audit)
  - manifest.json

No LLM, no JVM. ``--no-validate`` skips polyglot (generation + string-coverage
only) so the generator can be exercised before the native extension is built.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import ddl as D  # noqa: E402
from aegir.ontology.complex import FamilyComplex  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

logger = logging.getLogger("build-ddl-spine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-dir", default="src/aegir/ontology/catalog")
    p.add_argument("--family-complex", default="src/aegir/ontology/family_complex.json")
    p.add_argument("--dialects", nargs="+", default=list(D.DEFAULT_DIALECTS))
    p.add_argument("--per-family", type=int, default=None,
                   help="Optional cap on templates per family (default: all 540)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip polyglot validation + AST coverage (generation only)")
    p.add_argument("--no-materialize-rows", action="store_true",
                   help="Skip correct-by-construction RI-true row + view materialization (Track A)")
    p.add_argument("--no-emit-views", action="store_true",
                   help="Materialize base rows but skip the corpus views")
    p.add_argument("--row-seed", type=lambda x: int(x, 0), default=0xAE61,
                   help="deterministic row-synthesis seed (hex ok, e.g. 0xAE61)")
    p.add_argument("--output-dir",
                   default="/raid/checkpoints/aegir-artifacts/ddl_spine_v0/")
    return p.parse_args()


def catalog_files(catalog_dir: Path) -> list[Path]:
    return sorted(p for p in catalog_dir.glob("0*.json")
                  if ".candidate" not in p.name and "combined" not in p.name)


def compute_run_id(args: argparse.Namespace, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(f"dialects:{','.join(sorted(args.dialects))}\n".encode())
    h.update(f"per_family:{args.per_family}\n".encode())
    h.update(f"row_seed:{args.row_seed}\n".encode())
    h.update(f"materialize:{not args.no_materialize_rows}\n".encode())
    for p in sorted(files):
        h.update(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}\n".encode())
    return h.hexdigest()[:16]


def load_spine(files: list[Path], per_family: int | None) -> list[D.SpineTable]:
    spine: list[D.SpineTable] = []
    for path in files:
        family = path.stem
        cat = load_catalog(path)
        templates = cat.templates[:per_family] if per_family else cat.templates
        for t in templates:
            spine.append(D.template_to_table(t, family))
    return spine


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    files = catalog_files(REPO / args.catalog_dir if not Path(args.catalog_dir).is_absolute()
                          else Path(args.catalog_dir))
    if not files:
        logger.error("no catalog family files found under %s", args.catalog_dir)
        return 1

    run_id = compute_run_id(args, files)
    out = Path(args.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    created_at = _dt.datetime.now(_dt.timezone.utc)
    logger.info("run_id=%s  output=%s  dialects=%s  validate=%s",
                run_id, out, args.dialects, not args.no_validate)

    spine = load_spine(files, args.per_family)
    logger.info("lowered %d templates across %d families", len(spine), len(files))

    fc = FamilyComplex.from_json(REPO / args.family_complex if not Path(args.family_complex).is_absolute()
                                 else Path(args.family_complex))
    fk_edges, fk_audit = D.cross_family_fks(spine, fc)
    fks_by_src: dict[str, list] = {}
    for e in fk_edges:
        fks_by_src.setdefault(e.src_table, []).append(e)
    n_emitted = sum(1 for a in fk_audit if a.get("status") == "emitted")
    logger.info("cross-family FKs: %d emitted / %d candidates", n_emitted, len(fk_audit))

    # ── Track A: correct-by-construction RI-true rows + corpus views ──────────────
    ri_ok = None
    base_row_rows: list[dict] = []
    base_index_rows: list[dict] = []
    view_out_rows: list[dict] = []
    n_views_valid = 0
    if not args.no_materialize_rows:
        from aegir.ontology.chapter_tables import (build_views, definitions_for_spine,
                                                    entity_pools_for_spine)
        from aegir.ontology.rows import assert_referential_integrity, materialize_rows
        materialize_rows(spine, fk_edges, seed=args.row_seed, definitions=definitions_for_spine(spine),
                         entity_pools=entity_pools_for_spine(spine))
        try:
            assert_referential_integrity(spine, fk_edges)
            ri_ok = True
        except AssertionError as e:
            ri_ok = False
            logger.error("REFERENTIAL INTEGRITY VIOLATION: %s", e)
        fk_src: dict[str, dict[str, str]] = {}
        for e in fk_edges:
            fk_src.setdefault(e.src_table, {})[e.src_col] = e.dst_table
        for st in spine:
            tfk = fk_src.get(st.table.name, {})
            for ri, row in enumerate(st.table.rows):
                for ci, c in enumerate(st.table.columns):
                    base_row_rows.append({
                        "table_name": st.table.name, "row_ix": ri, "col_name": c.name,
                        "value": row[ci] if ci < len(row) else "",
                        "is_pk": c.slot_ref == "__pk__", "is_fk": c.name in tfk,
                        "fk_target_table": tfk.get(c.name), "run_id": run_id, "created_at": created_at})
            base_index_rows.append({
                "template_id": st.template.template_id, "table_name": st.table.name,
                "family": st.family, "n_rows": len(st.table.rows), "n_cols": len(st.table.columns),
                "columns_json": json.dumps([c.name for c in st.table.columns]),
                "run_id": run_id, "created_at": created_at})
        if not args.no_emit_views:
            for v, vrows in build_views(spine, fk_edges):
                valid = None
                if not args.no_validate:
                    valid = all(dv.valid for dv in D.validate_ddl(v.sql, tuple(args.dialects)))
                    n_views_valid += int(bool(valid))
                view_out_rows.append({
                    "view_name": v.name, "sql": v.sql, "verbalization": v.verbalization,
                    "base_tables_json": json.dumps(v.base_tables),
                    "columns_json": json.dumps([{"view_col": vc, "base_table": bt, "base_col": bc}
                                                for vc, bt, bc in v.columns]),
                    "fk_json": json.dumps(None if v.fk is None
                                          else {"src_col": v.fk.src_col, "dst_table": v.fk.dst_table}),
                    "n_rows": len(vrows), "rows_json": json.dumps(vrows), "valid": valid,
                    "run_id": run_id, "created_at": created_at})
        logger.info("materialized %d cells / %d tables; RI=%s; views=%d%s",
                    len(base_row_rows), len(spine), ri_ok, len(view_out_rows),
                    "" if args.no_validate else f" (valid={n_views_valid}/{len(view_out_rows)})")

    stmt_rows, val_rows = [], []
    feat_counts: Counter = Counter()      # (feature, variant) -> n tables with it
    feat_totals: Counter = Counter()      # (feature, variant) -> total occurrences
    n_valid_all = 0

    dialects = tuple(args.dialects)
    for st in spine:
        fks = fks_by_src.get(st.table.name, [])
        canonical = D.render_ddl(st, fks)
        iceberg = D.render_ddl(st, fks, iceberg=True)

        stmt_rows.append({
            "template_id": st.template.template_id, "table_name": st.table.name,
            "family": st.family, "bfo_anchor": list(st.template.bfo_anchor_path),
            "is_complex": bool(st.template.is_complex),
            "n_columns": len(st.table.columns), "n_fk": len(fks),
            "ddl_text": canonical, "ddl_iceberg": iceberg,
            "columns_json": json.dumps(D.column_dicts(st)),
            "fks_json": json.dumps([{"src_col": e.src_col, "dst_table": e.dst_table,
                                     "dst_col": e.dst_col} for e in fks]),
            "run_id": run_id, "created_at": created_at,
        })

        for variant, text, dlist in (("canonical", canonical, dialects),
                                     ("iceberg", iceberg, (D.ICEBERG_DIALECT,))):
            for feat, n in D.coverage_inventory(text, dialect=dlist[0],
                                                parse=not args.no_validate).items():
                feat_counts[(feat, variant)] += 1
                feat_totals[(feat, variant)] += n
            if not args.no_validate:
                for dv in D.validate_ddl(text, dlist):
                    val_rows.append({
                        "template_id": st.template.template_id, "table_name": st.table.name,
                        "variant": variant, "dialect": dv.dialect, "valid": dv.valid,
                        "n_errors": len(dv.errors),
                        "errors_json": json.dumps([e.__dict__ for e in dv.errors]),
                        "run_id": run_id, "created_at": created_at,
                    })

        if not args.no_validate:
            ok = all(dv.valid for dv in D.validate_ddl(canonical, dialects))
            n_valid_all += int(ok)

    _write(out / "ddl_statements.parquet", stmt_rows, {
        "template_id": pa.string(), "table_name": pa.string(), "family": pa.string(),
        "bfo_anchor": pa.list_(pa.string()), "is_complex": pa.bool_(),
        "n_columns": pa.int32(), "n_fk": pa.int32(), "ddl_text": pa.string(),
        "ddl_iceberg": pa.string(), "columns_json": pa.string(), "fks_json": pa.string(),
        "run_id": pa.string(), "created_at": pa.timestamp("us", tz="UTC"),
    })
    if val_rows:
        _write(out / "ddl_validation.parquet", val_rows, {
            "template_id": pa.string(), "table_name": pa.string(), "variant": pa.string(),
            "dialect": pa.string(), "valid": pa.bool_(), "n_errors": pa.int32(),
            "errors_json": pa.string(), "run_id": pa.string(),
            "created_at": pa.timestamp("us", tz="UTC"),
        })
    cov_rows = [{"feature": f, "variant": v, "n_tables": feat_counts[(f, v)],
                 "total_occurrences": feat_totals[(f, v)], "run_id": run_id,
                 "created_at": created_at}
                for (f, v) in sorted(feat_counts)]
    _write(out / "sql_feature_coverage.parquet", cov_rows, {
        "feature": pa.string(), "variant": pa.string(), "n_tables": pa.int32(),
        "total_occurrences": pa.int32(), "run_id": pa.string(),
        "created_at": pa.timestamp("us", tz="UTC"),
    })
    for a in fk_audit:
        a["run_id"] = run_id
        a["created_at"] = created_at
        a["simplex"] = a.get("simplex") or []
    _write(out / "cross_family_fks.parquet", fk_audit, {
        "src": pa.string(), "dst": pa.string(), "src_family": pa.string(),
        "dst_family": pa.string(), "via": pa.string(), "simplex": pa.list_(pa.string()),
        "allowed": pa.bool_(), "status": pa.string(), "run_id": pa.string(),
        "created_at": pa.timestamp("us", tz="UTC"),
    })
    if base_row_rows:
        _write(out / "base_rows.parquet", base_row_rows, {
            "table_name": pa.string(), "row_ix": pa.int32(), "col_name": pa.string(),
            "value": pa.string(), "is_pk": pa.bool_(), "is_fk": pa.bool_(),
            "fk_target_table": pa.string(), "run_id": pa.string(),
            "created_at": pa.timestamp("us", tz="UTC"),
        })
        _write(out / "base_table_index.parquet", base_index_rows, {
            "template_id": pa.string(), "table_name": pa.string(), "family": pa.string(),
            "n_rows": pa.int32(), "n_cols": pa.int32(), "columns_json": pa.string(),
            "run_id": pa.string(), "created_at": pa.timestamp("us", tz="UTC"),
        })
    if view_out_rows:
        _write(out / "views.parquet", view_out_rows, {
            "view_name": pa.string(), "sql": pa.string(), "verbalization": pa.string(),
            "base_tables_json": pa.string(), "columns_json": pa.string(), "fk_json": pa.string(),
            "n_rows": pa.int32(), "rows_json": pa.string(), "valid": pa.bool_(),
            "run_id": pa.string(), "created_at": pa.timestamp("us", tz="UTC"),
        })

    manifest = {
        "run_id": run_id, "created_at": created_at.isoformat(),
        "catalog_files": [p.name for p in files], "n_templates": len(spine),
        "dialects": list(dialects), "per_family": args.per_family,
        "validated": not args.no_validate,
        "cross_family_fks_emitted": n_emitted,
        "valid_all_dialects": None if args.no_validate else n_valid_all,
        "materialized_rows": not args.no_materialize_rows,
        "row_seed": args.row_seed,
        "referential_integrity_ok": ri_ok,
        "n_base_cells": len(base_row_rows),
        "n_views": len(view_out_rows),
        "views_valid": None if (args.no_validate or args.no_emit_views) else n_views_valid,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("DONE  templates=%d  features=%d  fks=%d%s",
                len(spine), len(feat_counts), n_emitted,
                "" if args.no_validate
                else f"  valid(trino∩spark)={n_valid_all}/{len(spine)}")
    return 0


def _write(path: Path, rows: list[dict], schema_fields: dict) -> None:
    schema = pa.schema(list(schema_fields.items()))
    cols = {name: [r.get(name) for r in rows] for name in schema_fields}
    pq.write_table(pa.table(cols, schema=schema), path)
    logger.info("wrote %s (%d rows)", path.name, len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
