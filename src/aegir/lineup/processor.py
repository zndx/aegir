"""ScheduledTaskProcessor — consume ``scheduled_tasks``, dispatch by ``task_type``.

The aegir port of Gaius's ``engine/services/scheduled_task_processor.py``: a
psycopg3-async LISTEN/NOTIFY + poll loop that claims pending rows
(``FOR UPDATE SKIP LOCKED``), runs the registered handler
(``aegir.lineup.maintain.HANDLERS``) off the event loop, and marks the row
completed/failed. pg_cron *enqueues* (``aegir.lineup.cron``); this *executes*.

Folded into the gateway (no separate process); guarded so a DB outage never stops
the gateway serving. ``process_pending`` is a single claim-and-dispatch batch — unit-
testable without the loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

log = logging.getLogger("aegir.lineup.processor")
CHANNEL = "aegir_scheduled_tasks"


def conninfo(db_url: str) -> str:
    """SQLAlchemy ``postgresql+psycopg://…`` → psycopg conninfo ``postgresql://…``."""
    return db_url.replace("+psycopg", "", 1)


class ScheduledTaskProcessor:
    def __init__(self, db_url: str, handlers: dict[str, Callable[[dict], dict]] | None = None,
                 poll_interval: float = 60.0):
        self.conninfo = conninfo(db_url)
        if handlers is None:
            from aegir.lineup import maintain
            handlers = maintain.HANDLERS
        self.handlers = dict(handlers)
        self.poll_interval = poll_interval
        self._stop = asyncio.Event()

    def register(self, task_type: str, handler: Callable[[dict], dict]) -> None:
        self.handlers[task_type] = handler

    async def process_pending(self, aconn, limit: int = 10) -> int:
        """Claim a batch of due pending tasks (SKIP LOCKED), run + mark each. Returns
        the number processed. Autocommit: each statement is its own transaction, so the
        single-statement claim is atomic and safe across concurrent processors."""
        async with aconn.cursor() as cur:
            await cur.execute(
                "UPDATE scheduled_tasks SET status='running', started_at=now() "
                "WHERE id IN (SELECT id FROM scheduled_tasks "
                "             WHERE status='pending' AND scheduled_for <= now() "
                "             ORDER BY scheduled_for FOR UPDATE SKIP LOCKED LIMIT %s) "
                "RETURNING id, task_type, payload", (limit,))
            claimed = await cur.fetchall()
        for tid, task_type, payload in claimed:
            handler = self.handlers.get(task_type)
            try:
                if handler is None:
                    raise KeyError(f"no handler for task_type {task_type!r}")
                result = await asyncio.to_thread(handler, payload or {})
                async with aconn.cursor() as cur:
                    await cur.execute(
                        "UPDATE scheduled_tasks SET status='completed', completed_at=now(), result=%s WHERE id=%s",
                        (json.dumps(result), tid))
                log.info("task %s (%s) ok: %s", tid, task_type, result)
            except Exception as e:                                    # noqa: BLE001 — record + continue
                log.warning("task %s (%s) failed: %s", tid, task_type, e)
                async with aconn.cursor() as cur:
                    await cur.execute(
                        "UPDATE scheduled_tasks SET status='failed', completed_at=now(), error=%s WHERE id=%s",
                        (str(e)[:500], tid))
        return len(claimed)

    async def run(self) -> None:
        import psycopg
        try:
            aconn = await psycopg.AsyncConnection.connect(self.conninfo, autocommit=True)
        except Exception as e:                                        # noqa: BLE001
            log.warning("processor disabled — cannot connect (%s); gateway still serves.", type(e).__name__)
            return
        log.info("ScheduledTaskProcessor up: LISTEN %s, handlers=%s", CHANNEL, sorted(self.handlers))
        async with aconn:
            await aconn.execute(f"LISTEN {CHANNEL}")
            while not self._stop.is_set():
                try:
                    await self.process_pending(aconn)
                except Exception as e:                                # noqa: BLE001
                    log.warning("process_pending error: %s", e)
                try:                                                  # wake on NOTIFY, else poll
                    async for _ in aconn.notifies(timeout=self.poll_interval, stop_after=1):
                        pass
                except TypeError:                                     # psycopg <3.2 — no timeout kwarg
                    await asyncio.sleep(self.poll_interval)
                except Exception:                                     # noqa: BLE001
                    await asyncio.sleep(min(self.poll_interval, 5))

    def stop(self) -> None:
        self._stop.set()
