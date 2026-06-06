"""aegir_hx — Apache AGE graph adapter: the read/write surface over the provenance graph.

A thin psycopg wrapper for Cypher against the ``aegir_hx`` graph (mirrors gaius's
``hx/lineage/graph.py``). Nodes/edges are Atlas-entity-aligned, so the same store
backs Atlas v2 entities + classifications AND OpenLineage runs/datasets — this module
is the functional surface the projector writes through and the UI reads from.

No fallback: requires AGE (the bootstrap guarantees it). Connects via the standard
PG* env (devenv sets these to :5555).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

GRAPH = "aegir_hx"


def _conninfo() -> str:
    return (
        f"host={os.environ.get('PGHOST', '127.0.0.1')} "
        f"port={os.environ.get('PGPORT', '5555')} "
        f"dbname={os.environ.get('PGDATABASE', 'aegir')} "
        f"user={os.environ.get('PGUSER', os.environ.get('USER', 'postgres'))}"
    )


@contextmanager
def connect():
    """Yield an autocommit psycopg connection with AGE loaded + search_path set."""
    import psycopg
    conn = psycopg.connect(_conninfo(), autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute('SET search_path = ag_catalog, "$user", public;')
        yield conn
    finally:
        conn.close()


# ── Cypher literal helpers (inline; values here are ids/scores/short labels) ──

def _lit(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def _props(d: dict | None) -> str:
    if not d:
        return "{}"
    return "{" + ", ".join(f"{k}: {_lit(v)}" for k, v in d.items()) + "}"


def _parse(v):
    """agtype text → python (strip ::vertex/::edge/::path suffixes, then JSON)."""
    if not isinstance(v, str):
        return v
    for suf in ("::vertex", "::edge", "::path"):
        if v.endswith(suf):
            v = v[: -len(suf)]
    try:
        return json.loads(v)
    except Exception:
        return v


# ── Core ops ─────────────────────────────────────────────────────────────────

def run(conn, cypher: str) -> list:
    """Execute a Cypher statement that RETURNs at least one column; returns parsed rows."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM cypher('{GRAPH}', $q${cypher}$q$) AS (v agtype);")
        return [_parse(r[0]) for r in (cur.fetchall() or [])]


def merge_node(conn, label: str, keys: dict, props: dict | None = None) -> None:
    """Idempotent upsert: MERGE on ``keys``, then SET the (mutable) ``props``."""
    set_clause = f" SET n += {_props(props)}" if props else ""
    run(conn, f"MERGE (n:{label} {_props(keys)}){set_clause} RETURN 1")


def merge_edge(conn, src_label: str, src_keys: dict, edge: str,
               dst_label: str, dst_keys: dict, props: dict | None = None) -> None:
    """Idempotent edge upsert between two already-keyed nodes."""
    set_clause = f" SET r += {_props(props)}" if props else ""
    run(conn,
        f"MATCH (a:{src_label} {_props(src_keys)}), (b:{dst_label} {_props(dst_keys)}) "
        f"MERGE (a)-[r:{edge}]->(b){set_clause} RETURN 1")


def scalar(conn, cypher: str):
    """Run a Cypher query expected to return a single row/column; return that value."""
    rows = run(conn, cypher)
    return rows[0] if rows else None
