"""lineage_views — panel-shaped materialized lineage (RH 2026-07-21).

Rendering Lineage straight from Atlas/AGE is slow: every ego render runs two live cypher()
scans (id() filters don't reach indexes through the wrapper). These are OUR materialized
views — plain indexed tables in aegir_hx, panel-shaped (display name precomputed, __rdbms
plumbing excluded), refreshed transactionally:

  lineage_nodes(vid PK, label, name)         lineage_edges(src, dst, etype) + indexes
  lineage_meta(refreshed_at, n_nodes, n_edges)

Ego reads become two indexed selects (O(degree)). Refresh paths:
  * `python -m aegir.lineup maintain lineage-refresh` — the upkeep-spine op
    (pg_cron → scheduled_tasks → in-gateway processor, like kb_archive/kb_snapshot)
  * throttled post-ingest refresh (ol.ingest_run_event fires a background refresh at most
    once per THROTTLE_S) — panels stay near-live without per-click graph scans
  * the gateway falls back to live cypher when the views are absent/empty (fresh DB) and
    triggers an ensure+refresh in the background.
"""
from __future__ import annotations

import threading
import time

from aegir.governance import graph as G

THROTTLE_S = 120
_last_refresh = [0.0]
_lock = threading.Lock()

_DISP_KEYS = ("name", "title", "chapter_id", "template_id", "run_id", "dataset", "qualifiedName")


def _disp(label: str, mp: "dict | None") -> str:
    mp = mp or {}
    for k in _DISP_KEYS:
        if mp.get(k):
            return str(mp[k])
    if mp.get("topic_id") is not None:
        return f"topic {mp['topic_id']}"
    return label


def ensure(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lineage_nodes (
                vid BIGINT PRIMARY KEY, label TEXT NOT NULL, name TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lineage_edges (
                src BIGINT NOT NULL, dst BIGINT NOT NULL, etype TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS lineage_edges_src ON lineage_edges (src);
            CREATE INDEX IF NOT EXISTS lineage_edges_dst ON lineage_edges (dst);
            CREATE TABLE IF NOT EXISTS lineage_meta (
                id INT PRIMARY KEY DEFAULT 1, refreshed_at TIMESTAMPTZ,
                n_nodes BIGINT, n_edges BIGINT);
        """)
    conn.commit()


def refresh() -> dict:
    """Full rebuild inside one transaction — the graph is modest; correctness beats
    incrementality until it is not."""
    with G.connect() as conn:
        ensure(conn)
        nodes = G.run(conn, "MATCH (n) RETURN {vid: id(n), label: labels(n)[0], "
                            "mp: properties(n)}")
        edges = G.run(conn, "MATCH (n)-[r]->(m) RETURN {src: id(n), dst: id(m), "
                            "etype: type(r)}")
        n_rows = [(int(x["vid"]), str(x.get("label") or "vertex"),
                   _disp(str(x.get("label") or "vertex"), x.get("mp")))
                  for x in nodes if str(x.get("label")) != "vertex"]
        e_rows = [(int(x["src"]), int(x["dst"]), str(x["etype"]))
                  for x in edges if not str(x.get("etype", "")).startswith("__rdbms")]
        with conn.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute("TRUNCATE lineage_nodes, lineage_edges")
            cur.executemany("INSERT INTO lineage_nodes VALUES (%s, %s, %s) "
                            "ON CONFLICT (vid) DO NOTHING", n_rows)
            cur.executemany("INSERT INTO lineage_edges VALUES (%s, %s, %s)", e_rows)
            cur.execute("INSERT INTO lineage_meta (id, refreshed_at, n_nodes, n_edges) "
                        "VALUES (1, now(), %s, %s) ON CONFLICT (id) DO UPDATE SET "
                        "refreshed_at = now(), n_nodes = EXCLUDED.n_nodes, "
                        "n_edges = EXCLUDED.n_edges", (len(n_rows), len(e_rows)))
            cur.execute("COMMIT")
    _last_refresh[0] = time.time()
    return {"n_nodes": len(n_rows), "n_edges": len(e_rows)}


def refresh_throttled_async() -> bool:
    """Post-ingest hook: at most one background refresh per THROTTLE_S. Never raises."""
    with _lock:
        if time.time() - _last_refresh[0] < THROTTLE_S:
            return False
        _last_refresh[0] = time.time()

    def _go():
        try:
            refresh()
        except Exception:  # noqa: BLE001 — a down graph must never break an ingest caller
            pass

    threading.Thread(target=_go, daemon=True).start()
    return True


def ego(focal: "int | None", cap: int = 40) -> "dict | None":
    """Indexed ego read. → None when the views are absent/empty (caller falls back live)."""
    try:
        with G.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT n_nodes FROM lineage_meta WHERE id = 1")
                row = cur.fetchone()
                if not row or not row[0]:
                    return None
                if focal is None:
                    cur.execute("SELECT vid FROM lineage_nodes WHERE label IN "
                                "('Dataset', 'Run', 'Chapter') ORDER BY label LIMIT 1")
                    r0 = cur.fetchone()
                    if not r0:
                        return {"focal": None, "neighbors": [], "truncated": False}
                    focal = int(r0[0])
                cur.execute("SELECT label, name FROM lineage_nodes WHERE vid = %s", (focal,))
                f = cur.fetchone()
                if not f:
                    return None
                cur.execute("""
                    SELECT m.vid, m.label, m.name, e.etype, e.src = %s AS out
                    FROM lineage_edges e
                    JOIN lineage_nodes m ON m.vid = CASE WHEN e.src = %s THEN e.dst ELSE e.src END
                    WHERE e.src = %s OR e.dst = %s
                    LIMIT %s""", (focal, focal, focal, focal, cap + 1))
                nb = cur.fetchall()
        truncated = len(nb) > cap
        nb = nb[:cap]
        return {"focal": {"vid": int(focal), "label": f[0], "name": f[1]},
                "neighbors": [{"vid": int(v), "label": l_, "name": n_, "edge": t_,
                               "out": bool(o_)} for v, l_, n_, t_, o_ in nb],
                "truncated": truncated, "cap": cap, "materialized": True}
    except Exception:  # noqa: BLE001
        return None
