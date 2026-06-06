"""Preflight + bootstrap for the aegir_hx Apache AGE provenance graph.

Ensures the graph substrate is loaded — idempotently and FAIL-LOUD. There is no
JSONB fallback: if AGE is not compiled into the running Postgres build, this
raises ``GraphSubstrateError`` with the remediation rather than degrading to a
weaker store. Same posture as the migration and the ColBERT MaxSim bridge.

The guarantee is layered; this module owns the bottom two and the preflight:

    build     extensions.age/pg_cron in devenv.nix      (direnv reload to realize)
    server    shared_preload_libraries = "age,pg_cron"  (devenv restart)
    database  CREATE EXTENSION + create_graph(aegir_hx) + labels   ← run_migrations()
    preflight assert_age_available() refuses a build without AGE   ← the no-fallback gate

Run before any graph operation (projector, Atlas sync) or standalone::

    uv run --no-sync python -m aegir.governance.bootstrap
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "migrations"
GRAPH = "aegir_hx"
REQUIRED = ("age", "pg_cron")


class GraphSubstrateError(RuntimeError):
    """The AGE substrate cannot be brought up — fail loud, no fallback."""


def _connect():
    """A psycopg(3)/psycopg2 connection from the standard PG* env (devenv sets these)."""
    conninfo = (
        f"host={os.environ.get('PGHOST', '127.0.0.1')} "
        f"port={os.environ.get('PGPORT', '5555')} "
        f"dbname={os.environ.get('PGDATABASE', 'aegir')} "
        f"user={os.environ.get('PGUSER', os.environ.get('USER', 'postgres'))}"
    )
    try:
        import psycopg
        return psycopg.connect(conninfo, autocommit=True)
    except ImportError:
        import psycopg2
        conn = psycopg2.connect(conninfo)
        conn.autocommit = True
        return conn


def assert_age_available(conn) -> None:
    """Refuse to proceed if AGE is not in the running build. No fallback."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM pg_available_extensions WHERE name = ANY(%s);", (list(REQUIRED),))
    have = {r[0] for r in cur.fetchall()}
    if "age" not in have:
        raise GraphSubstrateError(
            "Apache AGE is not in the running Postgres build (age.control absent).\n"
            "The graph-centric architecture has NO fallback — rebuild the devenv server:\n"
            "    direnv reload && devenv processes down && devenv up -d\n"
            "then re-run:  uv run --no-sync python -m aegir.governance.bootstrap"
        )
    if "pg_cron" not in have:
        log.warning("pg_cron unavailable — async materialization off (graph still works).")


def _up_block(sql_text: str) -> str:
    out, on = [], False
    for line in sql_text.splitlines():
        s = line.strip().lower()
        if s == "-- migrate:up":
            on = True
        elif s == "-- migrate:down":
            break
        elif on:
            out.append(line)
    return "\n".join(out)


def _statements(sql: str):
    body = "\n".join(ln for ln in sql.splitlines() if not ln.strip().startswith("--"))
    for stmt in body.split(";"):
        if stmt.strip():
            yield stmt.strip()


def run_migrations(conn, migrations_dir: Path = MIGRATIONS) -> list[str]:
    """Apply pending ``-- migrate:up`` blocks in filename order (dbmate-compatible
    tracking in ``schema_migrations``). LOAD/SET persist across statements within
    the one connection, so create_graph/create_vlabel resolve in ag_catalog."""
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version varchar(128) PRIMARY KEY);")
    cur.execute("SELECT version FROM schema_migrations;")
    applied = {r[0] for r in cur.fetchall()}
    ran: list[str] = []
    for f in sorted(migrations_dir.glob("*.sql")):
        if f.stem in applied:
            continue
        for stmt in _statements(_up_block(f.read_text())):
            cur.execute(stmt)
        cur.execute("INSERT INTO schema_migrations (version) VALUES (%s);", (f.stem,))
        ran.append(f.stem)
        log.info("applied migration %s", f.stem)
    return ran


def verify_graph(conn, graph: str = GRAPH) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s;", (graph,))
    if not cur.fetchone()[0]:
        raise GraphSubstrateError(f"graph {graph!r} missing after migration — bootstrap failed.")
    cur.execute(
        "SELECT count(*) FROM ag_catalog.ag_label "
        "WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s);", (graph,))
    return {"graph": graph, "labels": cur.fetchone()[0]}


def bootstrap() -> dict:
    """Preflight + ensure the aegir_hx substrate. Idempotent; raises loud on failure."""
    conn = _connect()
    try:
        assert_age_available(conn)
        ran = run_migrations(conn)
        info = verify_graph(conn)
        info["migrations_applied"] = ran
        log.info("aegir_hx ready: %s", info)
        return info
    finally:
        conn.close()


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(bootstrap(), indent=2))
    except GraphSubstrateError as e:
        print(f"\nBOOTSTRAP FAILED (no fallback by design):\n{e}")
        raise SystemExit(1)
