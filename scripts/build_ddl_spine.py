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
    p.add_argument("--realize", action="store_true",
                   help="realize each template into a schema SUBGRAPH (EAV / junction / star / snowflake) "
                        "instead of one flat table — the super-linear DDL+views deliverable. The profile is "
                        "DETERMINISTIC from the template's provenance.grounds_ddl (no sampling)")
    p.add_argument("--realize-seed", type=lambda x: int(x, 0), default=0x5EED,
                   help="retained for compatibility; profiles are deterministic (provenance.grounds_ddl), "
                        "the seed no longer selects structure")
    p.add_argument("--naming", choices=["natural", "semantic"], default="natural",
                   help="physical-name register (Convert 1c). 'natural' (default, the CANONICAL "
                        "deliverable) threads natural_names.json into schema realization so the whole "
                        "subgraph — sub-tables, FK columns, views, SQL — is constructed in the DBA "
                        "register from birth; 'semantic' keeps ontology-native names (Atlas/lineage "
                        "reference). Either way naming_map.parquet records the per-column "
                        "semantic↔natural pairing + name_provenance")
    p.add_argument("--output-dir",
                   default="/raid/checkpoints/aegir-artifacts/ddl_spine_v0/")
    return p.parse_args()


def catalog_files(catalog_dir: Path) -> list[Path]:
    from aegir.ontology.schema import catalog_files as _cf
    return _cf(catalog_dir)


def compute_run_id(args: argparse.Namespace, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(f"dialects:{','.join(sorted(args.dialects))}\n".encode())
    h.update(f"per_family:{args.per_family}\n".encode())
    h.update(f"row_seed:{args.row_seed}\n".encode())
    h.update(f"materialize:{not args.no_materialize_rows}\n".encode())
    h.update(f"realize:{args.realize}:{args.realize_seed}\n".encode())
    if args.naming != "semantic":  # register + the map content shape the run (legacy ids unchanged)
        from aegir.ontology.natural_naming import _RESOURCE as _NN
        nn_hash = hashlib.sha256(_NN.read_bytes()).hexdigest()[:16] if _NN.exists() else "absent"
        h.update(f"naming:{args.naming}:{nn_hash}\n".encode())
    for p in sorted(files):
        h.update(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}\n".encode())
    return h.hexdigest()[:16]


def load_spine(files: list[Path], per_family: int | None, *, realize: bool = False,
               realize_seed: int = 0x5EED, natural_names: dict | None = None):
    """Lower templates to the spine. With ``realize``, each template becomes a stochastic schema
    SUBGRAPH (realize.realize_schema) — returns (tables, intra-subgraph FKs, reconstruction views,
    per-template complexity rows). Flat mode returns one table/template (empty fk/view/cx lists).
    ``natural_names`` (Convert 1c) threads each template's natural record into construction so the
    subgraph is born in the DBA register (flat mode renames post-hoc via natural_naming — base-only,
    the proven path)."""
    nn = natural_names or {}
    spine: list[D.SpineTable] = []
    rfks: list = []
    rviews: list = []
    cx: list[dict] = []
    n_templates = 0
    all_ft: list = []
    for path in files:
        family = path.stem
        cat = load_catalog(path)
        templates = cat.templates[:per_family] if per_family else cat.templates
        for t in templates:
            n_templates += 1
            if realize:
                all_ft.append((family, t))
            else:
                spine.append(D.template_to_table(t, family))
    if realize:
        # C2.5 multi-template composition (aegir.ontology.compose): a domain-grouped 3NF core
        # (subgraphs + enum lookups) + a denorm layer of wide composite workbench tables. The
        # naming-map twin is built by re-calling this with natural_names={} — identical structure.
        from aegir.ontology import compose as cz
        from aegir.ontology.chapter_tables import definitions_for_spine
        spine, rfks, rviews, cx = cz.build_realized_spine(
            all_ft, natural_names=nn, realize_seed=realize_seed, definitions_fn=definitions_for_spine)
    elif nn:
        from aegir.ontology.natural_naming import natural_rename
        spine, _ = natural_rename(spine, [], nn)
    return spine, rfks, rviews, cx, n_templates


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

    natural_names: dict = {}
    if args.naming == "natural":
        from aegir.ontology.natural_naming import load_natural_names
        natural_names = load_natural_names()   # globally disambiguated (collisions → _2, _3 …)
        if not natural_names:
            logger.warning("--naming natural but natural_names.json is absent/empty — "
                           "falling back to the semantic register (run seed_natural_names.py)")
    spine, realized_fks, realized_views, complexity_rows, n_templates = load_spine(
        files, args.per_family, realize=args.realize, realize_seed=args.realize_seed,
        natural_names=natural_names)
    logger.info("lowered %d templates → %d tables (%s, %s register) across %d families",
                n_templates, len(spine), "realized subgraphs" if args.realize else "flat",
                args.naming if natural_names or args.naming == "semantic" else "semantic(fallback)",
                len(files))
    # spine-wide physical-name uniqueness (natural table names are globally disambiguated at load;
    # sub-table names inherit uniqueness from their base prefix — this asserts the invariant)
    _names_seen: dict[str, str] = {}
    for st in spine:
        prev = _names_seen.setdefault(st.table.name, st.template.template_id)
        if prev != st.template.template_id:
            logger.error("TABLE NAME COLLISION: %s (templates %s / %s)", st.table.name, prev,
                         st.template.template_id)
            return 1

    # ── naming map (Convert 1c): the per-column semantic↔natural lineage edge + name provenance ────
    # Built by realizing the SEMANTIC twin and zipping positionally (construction is deterministic, so
    # the twin has identical structure — which this also asserts: a register must never change shape).
    naming_rows: list[dict] = []
    if natural_names:
        sem_spine = load_spine(files, args.per_family, realize=args.realize,
                               realize_seed=args.realize_seed)[0]
    else:
        sem_spine = spine
    if len(sem_spine) != len(spine):
        logger.error("register-twin mismatch: %d semantic vs %d %s tables", len(sem_spine), len(spine),
                     args.naming)
        return 1
    from aegir.ontology.natural_naming import provenance_of
    for sst, nst in zip(sem_spine, spine):
        if (sst.template.template_id != nst.template.template_id
                or len(sst.table.columns) != len(nst.table.columns)):
            logger.error("register-twin structural divergence at %s / %s", sst.table.name, nst.table.name)
            return 1
        rec = natural_names.get(nst.template.template_id) or {}
        rec_prov = provenance_of(rec) if rec else "semantic"
        rec_cols = set((rec.get("cols") or {}).values())
        # per-column SOURCE-template attribution: composite (denorm) tables union columns from many
        # templates, so a single per-table template_id can't attribute a column — realize_meta.col_source
        # carries the origin per position. Non-composite tables attribute every column to their template.
        col_source = nst.realize_meta.get("col_source") or []
        layer = nst.realize_meta.get("layer", "core")
        for ci, (sc, nc) in enumerate(zip(sst.table.columns, nst.table.columns)):
            prov = ("semantic-passthrough" if nc.name == sc.name
                    else rec_prov if nc.name in rec_cols else "composed")
            src_tid = (col_source[ci] if ci < len(col_source) and col_source[ci] != "__pk__"
                       else nst.template.template_id)
            naming_rows.append({
                "template_id": nst.template.template_id, "kind": nst.kind,
                "table_name": nst.table.name, "semantic_table": sst.table.name,
                "col_name": nc.name, "semantic_col": sc.name, "slot_ref": nc.slot_ref,
                "register": args.naming if natural_names else "semantic",
                "name_provenance": prov, "source_template_id": src_tid, "layer": layer})
    if naming_rows:
        from collections import Counter as _CP
        logger.info("naming map: %d columns, provenance %s", len(naming_rows),
                    dict(_CP(r["name_provenance"] for r in naming_rows)))

    # None since the family-complex retirement (co-occurrence is measured, never a pre-wired gate);
    # cross_family_fks treats None as no-sanction and still audits the candidates.
    fc = FamilyComplex.load_optional(REPO / args.family_complex if not Path(args.family_complex).is_absolute()
                                     else Path(args.family_complex))
    primary: list = []
    cross_fks: list = []
    if args.realize:
        # cross-family FKs link the PRIMARY entity table of each template's subgraph (one per template);
        # intra-subgraph FKs (EAV/junction/star) are already in realized_fks.
        primary = [st for st in spine if st.kind == "entity"
                   and st.table.name == D.table_name(st.template.template_id)]
        cross_fks, fk_audit = D.cross_family_fks(primary, fc)
        # a profile may move a restriction-target column out of the primary table (into a junction/dim),
        # so keep only cross-FKs whose endpoints still exist (RI + view-join safe).
        _cols = {st.table.name: {c.name for c in st.table.columns} for st in spine}
        cross_fks = [e for e in cross_fks
                     if e.src_col in _cols.get(e.src_table, set()) and e.dst_col in _cols.get(e.dst_table, set())]
        fk_edges = realized_fks + cross_fks
        logger.info("realized: %d intra-subgraph FKs + %d cross-family FKs", len(realized_fks), len(cross_fks))
    else:
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
    pool_src: dict = {}
    if not args.no_materialize_rows:
        from aegir.ontology.chapter_tables import (build_views, definitions_for_spine,
                                                    entity_pool_sources, entity_pools_for_spine)
        from aegir.ontology.rows import assert_referential_integrity, materialize_rows
        # pools are keyed by SEMANTIC column names — the natural map translates them onto the
        # constructed register (Convert 1c; without this every pooled value regresses to a placeholder)
        materialize_rows(spine, fk_edges, seed=args.row_seed, definitions=definitions_for_spine(spine),
                         entity_pools=entity_pools_for_spine(spine, natural_names))
        # the value-provenance audit (Convert 1b): which source fed each pooled template's entity cells —
        # registry (in-loop derived individuals, membrane-gated) vs legacy (frozen pool) vs unpooled
        # (curated/type generators). The static fraction should shrink toward 0 as the registry accretes.
        src_map = entity_pool_sources()
        from collections import Counter as _C2
        pool_src = _C2(src_map.get(st.template.template_id, "unpooled") for st in spine)
        logger.info("entity-value sources (tables): %s", dict(pool_src))
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
            # realize mode: the reconstruction views (EAV-long / junction-resolved / star-denorm) are emitted
            # by the realizer; cross-family views come from the primary entity tables. Flat mode: as before.
            if args.realize:
                view_pairs = [(v, []) for v in realized_views] + list(build_views(primary, cross_fks))
            else:
                view_pairs = build_views(spine, fk_edges)
            for v, vrows in view_pairs:
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
    if naming_rows:
        for r in naming_rows:
            r["run_id"] = run_id
            r["created_at"] = created_at
        _write(out / "naming_map.parquet", naming_rows, {
            "template_id": pa.string(), "kind": pa.string(), "table_name": pa.string(),
            "semantic_table": pa.string(), "col_name": pa.string(), "semantic_col": pa.string(),
            "slot_ref": pa.string(), "register": pa.string(), "name_provenance": pa.string(),
            "source_template_id": pa.string(), "layer": pa.string(),
            "run_id": pa.string(), "created_at": pa.timestamp("us", tz="UTC"),
        })

    realize_summary: dict = {}
    if complexity_rows:
        from collections import Counter as _C
        for r in complexity_rows:
            r["run_id"] = run_id
            r["created_at"] = created_at
        _write(out / "structural_complexity.parquet", complexity_rows, {
            "template_id": pa.string(), "family": pa.string(), "profile": pa.string(),
            "profile_source": pa.string(), "layer": pa.string(), "n_composed": pa.int32(),
            "n_tables": pa.int32(), "n_views": pa.int32(), "n_fks": pa.int32(),
            "eav_tables": pa.int32(), "junction_tables": pa.int32(), "attr_count": pa.int32(),
            "eav_ratio": pa.float64(), "m2n_density": pa.float64(), "fk_depth": pa.int32(),
            "dim_tables": pa.int32(), "run_id": pa.string(), "created_at": pa.timestamp("us", tz="UTC"),
        })
        tt = sum(r["n_tables"] for r in complexity_rows)
        n_denorm = sum(1 for r in complexity_rows if r.get("layer") == "denorm")
        realize_summary = {
            "profile_distribution": dict(_C(r["profile"] for r in complexity_rows)),
            # the authenticity audit: how each profile was CHOSEN (grounds_ddl:* = lowering-by-theorem;
            # default-minimal = no grounding signal, the work-queue; there is no sampled path)
            "profile_source_distribution": dict(_C(r.get("profile_source", "?") for r in complexity_rows)),
            # C2.5 composition strata: denorm composite workbench tables vs the 3NF core (subgraphs+lookups)
            "layer_distribution": dict(_C(r.get("layer", "?") for r in complexity_rows)),
            "denorm_composites": n_denorm,
            "denorm_members_composed": sum(r.get("n_composed", 0) or 0 for r in complexity_rows),
            "mean_composition_size": (round(sum(r.get("n_composed", 0) or 0 for r in complexity_rows) / n_denorm, 2)
                                      if n_denorm else 0.0),
            # and which source fed entity-cell values (registry = in-loop individuals; legacy = frozen pool)
            "value_pool_sources": dict(pool_src) if not args.no_materialize_rows else {},
            "total_tables": tt, "total_views": sum(r["n_views"] for r in complexity_rows),
            "mean_eav_ratio": round(sum(r["eav_ratio"] for r in complexity_rows) / len(complexity_rows), 3),
            "max_fk_depth": max(r["fk_depth"] for r in complexity_rows),
        }
        logger.info("realize complexity: %s", realize_summary)

    from collections import Counter as _CM
    manifest = {
        "run_id": run_id, "created_at": created_at.isoformat(),
        "catalog_files": [p.name for p in files], "n_templates": n_templates,
        "n_tables": len(spine),
        "naming": args.naming if natural_names or args.naming == "semantic" else "semantic(fallback)",
        "n_naturalized_templates": len(natural_names),
        "name_provenance_distribution": dict(_CM(r["name_provenance"] for r in naming_rows)),
        "dialects": list(dialects), "per_family": args.per_family,
        "validated": not args.no_validate,
        "cross_family_fks_emitted": n_emitted,
        "valid_all_dialects": None if args.no_validate else n_valid_all,
        "materialized_rows": not args.no_materialize_rows,
        "row_seed": args.row_seed,
        "realize": args.realize,
        "realize_summary": realize_summary,
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
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
