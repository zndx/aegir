"""Database bootstrap: apply SQL migrations, idempotent.

Pattern is a light port of Atelier's ``src/atelier/db/bootstrap.py``:

1. Ensure ``schema_migrations`` table exists (dbmate-compatible shape:
   single ``version VARCHAR`` column).
2. For each ``migrations/*.sql`` file, extract the ``-- migrate:up``
   block and apply it if the version hasn't already been recorded.
   (Tracked at top-level ``migrations/`` — ``db/`` is devenv runtime state.)
3. Skip keystone-data seeding — M1 has no agents/catalog to upsert;
   migration content is schema-only.

Run as a module, so CAI's ``bin/start-app.sh`` and devenv's ``processes``
block can call::

    python -m aegir.db.bootstrap

which reads the database URL from ``aegir.config.load_config().db.url``
(respecting env overrides).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ── Migration runner ────────────────────────────────────────────


def run_migrations(db_url: str, migrations_dir: str | Path = "migrations") -> None:
    """Apply SQL migrations from the migrations directory.

    Reads ``-- migrate:up`` blocks from migration files and executes them in
    filename order. Tracks applied versions in a ``schema_migrations`` table
    (dbmate-compatible). Idempotent — re-running is a no-op.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url, connect_args={"connect_timeout": 10})
    migrations_path = Path(migrations_dir)

    if not migrations_path.exists():
        log.warning("Migrations directory %s not found, skipping", migrations_dir)
        engine.dispose()
        return

    with engine.begin() as conn:
        _assert_age_available(conn)
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version VARCHAR(128) PRIMARY KEY"
            ")"
        ))

        result = conn.execute(text("SELECT version FROM schema_migrations"))
        applied = {row[0] for row in result}

        for sql_file in sorted(migrations_path.glob("*.sql")):
            version = sql_file.stem
            if version in applied:
                continue
            log.info("applying migration: %s", version)
            up_sql = _extract_up_block(sql_file.read_text())
            if not up_sql:
                log.warning("migration %s has no -- migrate:up block, skipping", version)
                continue
            for stmt in _split_statements(up_sql):
                # exec_driver_sql: raw to the driver — migration DDL must NOT be
                # subject to SQLAlchemy's ":name" bind-param parsing (Cypher/comments
                # legitimately contain colons, e.g. [:INPUT_TO], chapter:foo).
                conn.exec_driver_sql(stmt)
            conn.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:v)"),
                {"v": version},
            )
            log.info("applied: %s", version)

    engine.dispose()
    log.info("migrations complete")


def _assert_age_available(conn) -> None:
    """No-fallback gate: refuse to migrate if Apache AGE is absent from the build.

    aegir's provenance graph is AGE-only — there is no relational fallback, so a
    missing extension is a loud, actionable failure rather than silent degradation.
    """
    from sqlalchemy import text

    have = {r[0] for r in conn.execute(text(
        "SELECT name FROM pg_available_extensions WHERE name IN ('age', 'pg_cron')"))}
    if "age" not in have:
        raise RuntimeError(
            "Apache AGE is not in the running Postgres build (age.control absent).\n"
            "The graph-centric architecture has NO fallback — rebuild the devenv server:\n"
            "    direnv reload && devenv processes down && devenv up -d\n"
            "(watch postgresql-and-plugins recompile with age + pg_cron), then re-run\n"
            "    uv run --no-sync python -m aegir.db.bootstrap"
        )
    if "pg_cron" not in have:
        log.warning("pg_cron unavailable — async graph materialization disabled.")


def _extract_up_block(content: str) -> str | None:
    """Extract SQL between ``-- migrate:up`` and ``-- migrate:down`` markers."""
    lines = content.splitlines()
    in_up = False
    up_lines: list[str] = []
    for line in lines:
        s = line.strip().lower()
        if s == "-- migrate:up":
            in_up = True
            continue
        if s == "-- migrate:down":
            break
        if in_up:
            up_lines.append(line)
    sql = "\n".join(up_lines).strip()
    return sql if sql else None


def _split_statements(sql: str) -> list[str]:
    """Split SQL on semicolons, respecting ``--`` line comments, ``/* */``
    block comments, and single-quoted strings (including ``''`` escapes).

    Ported verbatim from ``atelier/src/atelier/db/bootstrap.py:101-168``
    because naive ``sql.split(';')`` broke on the first ``;`` inside a
    comment — a bug we don't need to re-discover.
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            buf.append(ch)
            i += 1
            while i < n and sql[i] != "\n":
                buf.append(sql[i])
                i += 1
            continue
        if ch == "/" and nxt == "*":
            buf.append(ch); buf.append(nxt)
            i += 2
            while i < n and not (sql[i] == "*" and i + 1 < n and sql[i + 1] == "/"):
                buf.append(sql[i])
                i += 1
            if i < n:
                buf.append("*"); buf.append("/")
                i += 2
            continue
        if ch == "'":
            buf.append(ch); i += 1
            while i < n:
                c = sql[i]; buf.append(c); i += 1
                if c == "'":
                    if i < n and sql[i] == "'":
                        buf.append("'"); i += 1
                        continue
                    break
            continue
        if ch == "$":
            # dollar-quoted body ($$…$$ or $tag$…$tag$): semicolons inside are literal,
            # so DO blocks and function bodies survive statement splitting.
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            if j < n and sql[j] == "$":
                tag = sql[i:j + 1]
                end = sql.find(tag, j + 1)
                if end == -1:
                    buf.append(sql[i:]); i = n
                else:
                    buf.append(sql[i:end + len(tag)]); i = end + len(tag)
                continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


# ── CLI entrypoint ──────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from aegir.config import load_config

    cfg = load_config()
    log.info("target db: %s", cfg.db.url.split("@")[-1] if "@" in cfg.db.url else cfg.db.url)
    run_migrations(cfg.db.url)


if __name__ == "__main__":
    main()
