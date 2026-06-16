"""Register the lineup-upkeep pg_cron jobs (``aegir.lineup install-cron``).

pg_cron lives in the ``postgres`` DB (``cron.database_name=postgres``), so the jobs are
registered THERE via ``cron.schedule_in_database(..., database := 'aegir')`` — they fire
against ``aegir`` (where ``scheduled_tasks`` + the processor live). The jobs only ENQUEUE
(or run pure-SQL cleanup); the Python handler does the filesystem work. Idempotent
(``cron.schedule*`` upserts by name). Guarded: a no-op + warning if pg_cron is absent, so
``devenv up`` never fails on it.
"""
from __future__ import annotations

import logging

log = logging.getLogger("aegir.lineup.cron")

# (job_name, cron_schedule, command executed in the `aegir` DB)
JOBS: list[tuple[str, str, str]] = [
    ("aegir-kb-archive", "30 3 * * *",                 # daily 03:30 — age past-quarter scratch → archive
     "INSERT INTO scheduled_tasks (task_type, source) VALUES ('kb_archive', 'pg_cron')"),
    ("aegir-kb-snapshot", "0 4 1 1,4,7,10 *",          # quarterly (1st of Jan/Apr/Jul/Oct, 04:00) — snapshot current/
     "INSERT INTO scheduled_tasks (task_type, source) VALUES ('kb_snapshot', 'pg_cron')"),
    ("aegir-scheduled-tasks-cleanup", "0 5 * * 0",     # weekly Sun 05:00 — prune old finished tasks (pure SQL)
     "DELETE FROM scheduled_tasks WHERE status IN ('completed','failed') "
     "AND completed_at < now() - interval '30 days'"),
]


def _postgres_conninfo(aegir_db_url: str) -> str:
    """The aegir conninfo with dbname swapped to ``postgres`` (where pg_cron lives)."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from aegir.lineup.processor import conninfo
    d = conninfo_to_dict(conninfo(aegir_db_url))
    d["dbname"] = "postgres"
    return make_conninfo(**d)


def install(aegir_db_url: str, aegir_db: str = "aegir") -> dict:
    """Register the upkeep jobs in the postgres DB, targeting the aegir DB. Idempotent."""
    import psycopg
    pg = _postgres_conninfo(aegir_db_url)
    try:
        with psycopg.connect(pg, autocommit=True, connect_timeout=10) as conn, conn.cursor() as cur:
            # pg_cron is preloaded (shared_preload_libraries) but must be CREATE'd in the
            # cron.database_name DB (= postgres). Idempotent; superuser, like db.bootstrap's age.
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_cron")
            for name, sched, cmd in JOBS:
                cur.execute("SELECT cron.schedule_in_database(%s, %s, %s, %s)", (name, sched, cmd, aegir_db))
    except Exception as e:                                  # noqa: BLE001 — never block startup on scheduling
        log.warning("lineup cron install skipped (%s: %s)", type(e).__name__, str(e)[:160])
        return {"installed": 0, "error": type(e).__name__}
    names = [j[0] for j in JOBS]
    log.info("lineup cron jobs registered in postgres → aegir: %s", names)
    return {"installed": len(names), "jobs": names}
