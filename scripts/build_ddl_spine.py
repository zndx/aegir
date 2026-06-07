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

    manifest = {
        "run_id": run_id, "created_at": created_at.isoformat(),
        "catalog_files": [p.name for p in files], "n_templates": len(spine),
        "dialects": list(dialects), "per_family": args.per_family,
        "validated": not args.no_validate,
        "cross_family_fks_emitted": n_emitted,
        "valid_all_dialects": None if args.no_validate else n_valid_all,
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
