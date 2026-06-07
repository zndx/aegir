"""Self-check for the ``polyglot_sql`` native extension (maturin/PyO3).

Run after building::

    devenv tasks run polyglot:build      # maturin develop --release
    uv run --no-sync python -m aegir.ontology._polyglot

Confirms the API surface the DDL spine + Atlas projector depend on is importable
and behaves, prints the supported dialects, and shows whether a sample CREATE
TABLE (with PK/FK/NOT NULL constraints) validates under Trino + Spark — which
determines whether the spine renders inline constraints or falls back to
columns-only. Exits non-zero with a build hint if the extension is missing.
"""

from __future__ import annotations

_REQUIRED = ["parse_one", "validate", "transpile", "lineage",
             "openlineage_run_event", "openlineage_column_lineage", "dialects"]

_SAMPLE = (
    "CREATE TABLE t_demo (\n"
    "  id VARCHAR(255),\n"
    "  x VARCHAR(255) NOT NULL,\n"
    "  y VARCHAR(255),\n"
    "  PRIMARY KEY (id),\n"
    "  FOREIGN KEY (y) REFERENCES t_other(id)\n"
    ") COMMENT 'aegir:self-check'"
)


def main() -> int:
    try:
        import polyglot_sql as pg
    except ImportError as e:
        print(f"FAIL: polyglot_sql not importable: {e}")
        print("  build it:  devenv tasks run polyglot:build")
        return 1

    missing = [f for f in _REQUIRED if not hasattr(pg, f)]
    if missing:
        print(f"FAIL: polyglot_sql missing expected API: {missing}")
        return 1

    dialects = pg.dialects()
    print(f"polyglot_sql OK — {len(dialects)} dialects")
    for d in ("trino", "spark"):
        print(f"  dialect {d!r}: {'present' if d in dialects else 'MISSING'}")

    ast = pg.parse_one(_SAMPLE, dialect="trino")
    print(f"  parse_one: kind={ast.kind!r}")
    for d in ("trino", "spark"):
        res = pg.validate(_SAMPLE, dialect=d)
        errs = getattr(res, "errors", None) or []
        print(f"  validate[{d}]: valid={res.valid} errors={len(errs)}"
              + (f" -> {errs[0].message}" if errs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
