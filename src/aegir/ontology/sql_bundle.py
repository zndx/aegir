"""sql_bundle — directly-loadable SQL projection of a released spine run (RH 2026-07-23).

The parquet artifacts remain the dataset of record; this DERIVES per-database loadable
SQL from them, partitioned one flavor per directory (``sql/postgres|trino|spark/``),
each self-contained: ``00_schema.sql`` (DDL), ``01_data.sql`` (RI-true multi-row
INSERTs from base_rows), ``02_views.sql``. postgres is transpiled from the trino
source via polyglot_sql (the 32-dialect transpiler already gating spine validation),
with the trino ``COMMENT`` payload re-emitted as first-class ``COMMENT ON TABLE`` —
the template/BFO provenance lands IN the loaded database. trino ships the native
``ddl_text``; spark ships ``ddl_iceberg``.

Determinism: derived from the parquets in stable order, no timestamps — re-emission
over an unchanged run is byte-stable (the release act stays no-op idempotent).
Failed transpiles are EXCLUDED AND COUNTED, never silently passed through.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DIALECTS = ("postgres", "trino", "spark")
_NUM_TYPES = ("INT", "BIGINT", "SMALLINT", "TINYINT", "DOUBLE", "FLOAT", "REAL",
              "DECIMAL", "NUMERIC")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
_COMMENT_RE = re.compile(r"\s*\nCOMMENT '((?:[^']|'')*)'\s*$")


def _lit(value, sql_type: str) -> str:
    if value is None:
        return "NULL"
    t = (sql_type or "").upper()
    if any(t.startswith(p) for p in _NUM_TYPES) and _NUM_RE.match(str(value)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _q(name: str, dialect: str) -> str:
    """Quote an identifier when the dialect requires it (digit-leading names — the
    spine's known validity gap, e.g. ``930nm_variant``): postgres/trino double-quote,
    spark backticks. Clean identifiers pass through bare."""
    if not name[:1].isdigit():
        return name
    return f"`{name}`" if dialect == "spark" else f'"{name}"'


def _pg():
    import polyglot_sql
    return polyglot_sql


def emit_sql_bundle(run_dir: "Path | str", dialects: "tuple[str, ...]" = DIALECTS) -> dict:
    """Write ``<run_dir>/sql/<dialect>/{00_schema,01_data,02_views}.sql`` (+ README).
    Returns per-dialect counts; raises nothing for a missing run artifact — the caller
    decides. Transpile failures are excluded and counted per dialect."""
    import pyarrow.parquet as pq
    run_dir = Path(run_dir)
    ddl = pq.read_table(run_dir / "ddl_statements.parquet").to_pylist()
    views = pq.read_table(run_dir / "views.parquet").to_pylist()
    base = pq.read_table(run_dir / "base_rows.parquet").to_pylist()
    poly = _pg()

    # column order + types per table, from the DDL's own record
    col_order: dict = {}
    col_type: dict = {}
    for r in ddl:
        cols = json.loads(r["columns_json"])
        col_order[r["table_name"]] = [c["name"] for c in cols]
        col_type[r["table_name"]] = {c["name"]: c.get("sql_type", "") for c in cols}

    # rows: (table, row_ix) → {col: value}; table order follows the DDL artifact
    cells: dict = {}
    for b in base:
        cells.setdefault(b["table_name"], {}).setdefault(int(b["row_ix"]), {})[
            b["col_name"]] = b["value"]

    def data_sql(dialect: str) -> "tuple[str, int]":
        out = []
        n_rows = 0
        for r in ddl:
            t = r["table_name"]
            trows = cells.get(t)
            if not trows:
                continue
            names = col_order[t]
            types = col_type[t]
            vals = []
            for ix in sorted(trows):
                row = trows[ix]
                vals.append("(" + ", ".join(_lit(row.get(c), types[c]) for c in names) + ")")
                n_rows += 1
            collist = ", ".join(_q(c, dialect) for c in names)
            out.append(f"INSERT INTO {_q(t, dialect)} ({collist}) VALUES\n  "
                       + ",\n  ".join(vals) + ";")
        return "\n\n".join(out) + "\n", n_rows

    summary: dict = {"run": run_dir.name, "dialects": {}}
    for d in dialects:
        ddir = run_dir / "sql" / d
        ddir.mkdir(parents=True, exist_ok=True)
        data_text, n_rows = data_sql(d)
        schema_stmts, comments, excluded = [], [], []
        for r in ddl:
            if d == "trino":
                schema_stmts.append(r["ddl_text"].rstrip() + ";")
                continue
            if d == "spark":
                schema_stmts.append(r["ddl_iceberg"].rstrip() + ";")
                continue
            # postgres: assemble the CREATE TABLE from the RECORDED columns_json
            # (identifiers quoted where the dialect requires), let polyglot map types,
            # and re-emit the trino COMMENT payload as COMMENT ON TABLE — the
            # template/BFO provenance becomes queryable in-database (obj_description()).
            m = _COMMENT_RE.search(r["ddl_text"])
            payload = m.group(1) if m else ""
            cols = json.loads(r["columns_json"])
            defs = [f"{_q(c['name'], d)} {c.get('sql_type', 'VARCHAR(255)')}"
                    f"{'' if c.get('nullable', True) else ' NOT NULL'}" for c in cols]
            pks = [_q(c["name"], d) for c in cols if c.get("pk")]
            bare = (f"CREATE TABLE {_q(r['table_name'], d)} (" + ", ".join(defs)
                    + (f", PRIMARY KEY ({', '.join(pks)})" if pks else "") + ")")
            try:
                stmts = poly.transpile(bare, read="trino", write="postgres")
                schema_stmts.append(";\n".join(s.rstrip() for s in stmts) + ";")
            except Exception:  # noqa: BLE001 — excluded AND counted, never silent
                excluded.append(r["table_name"])
                continue
            if payload:
                comments.append(f"COMMENT ON TABLE {r['table_name']} IS '{payload}';")
        # digit-leading identifiers lex as NUMBER+alias when unquoted (postgres AND
        # trino) — quote every known occurrence in view SQL before shipping/transpiling
        digit_ids = sorted({n for tps in col_type.values() for n in tps
                            if n[:1].isdigit()} |
                           {r["table_name"] for r in ddl
                            if r["table_name"][:1].isdigit()}, key=len, reverse=True)
        did_re = re.compile(r'(?<![\w"`])(' + "|".join(map(re.escape, digit_ids))
                            + r')(?![\w"`])') if digit_ids else None

        def _quote_ids(sql: str) -> str:
            return did_re.sub(lambda m: _q(m.group(1), d), sql) if did_re else sql

        view_stmts = []
        n_view_excluded = 0
        for v in views:
            if not v.get("valid", True):
                continue
            sql = _quote_ids(v["sql"].rstrip().rstrip(";"))
            if d == "trino":
                view_stmts.append(sql + ";")
                continue
            try:
                stmts = poly.transpile(sql, read="trino", write=d)
                view_stmts.append(";\n".join(s.rstrip() for s in stmts) + ";")
            except Exception:  # noqa: BLE001
                n_view_excluded += 1
        (ddir / "00_schema.sql").write_text(
            "\n\n".join(schema_stmts + comments) + "\n")
        (ddir / "01_data.sql").write_text(data_text)
        (ddir / "02_views.sql").write_text("\n\n".join(view_stmts) + "\n")
        summary["dialects"][d] = {
            "tables": len(schema_stmts), "insert_rows": n_rows,
            "views": len(view_stmts), "ddl_excluded": len(excluded),
            "views_excluded": n_view_excluded}
        if excluded:
            summary["dialects"][d]["excluded"] = excluded

    lines = [f"| {d} | {s['tables']} | {s['insert_rows']} | {s['views']} |"
             for d, s in summary["dialects"].items()]
    (run_dir / "sql" / "README.md").write_text(f"""# Loadable SQL — spine run `{run_dir.name}`

Derived from the parquet dataset of record in this directory (one flavor per
subdirectory — self-contained; load the files in numeric order):

| flavor | tables | rows | views |
|---|---|---|---|
{chr(10).join(lines)}

```sh
# PostgreSQL (see also `just load-postgres` at the repo root)
psql "$CONN" -v ON_ERROR_STOP=1 \\
  -f postgres/00_schema.sql -f postgres/01_data.sql -f postgres/02_views.sql
```

postgres DDL is transpiled from the trino source by polyglot_sql; each table's
template/BFO provenance ships as `COMMENT ON TABLE` (query it via
`obj_description('<table>'::regclass)`). trino is the native `ddl_text`; spark is
the Iceberg flavor (`ddl_iceberg`). Ontology↔table associations:
`../ontology_entity_associations.json`.
""")
    return summary
