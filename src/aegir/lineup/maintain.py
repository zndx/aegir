"""``aegir.lineup maintain`` — the KB upkeep handlers (the work the scheduler drives).

These are the task handlers the ScheduledTaskProcessor dispatches to (and runnable by
hand: ``python -m aegir.lineup maintain {upkeep|snapshot|reproject}``). Pure filesystem
(+ reproject = re-run the projection) — no DB; the scheduling/queue is separate
(pg_cron → scheduled_tasks → processor → here), the Gaius pattern.

  • upkeep    — move authored scratch notes (``scratch/<iso-date>/…``) whose date is
                BEFORE the current quarter into ``archive/<year>Q<n>/<iso-date>/…``.
                Current-quarter notes stay in scratch. Idempotent.
  • snapshot  — copy the projected ``current/`` → ``archive/<year>Q<n>/_snapshot_<utc>/``
                (the ontology as-it-was, for diffing over time).
  • reproject — re-run the projection (``build.run``) so ``current/`` tracks the catalog.
"""
from __future__ import annotations

import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from aegir.lineup import sources as S


def _quarter(d: date) -> int:
    return (d.month - 1) // 3 + 1


def _quarter_start(d: date) -> date:
    return date(d.year, 3 * (_quarter(d) - 1) + 1, 1)


def upkeep(kb_dir: str | Path | None = None, today: date | None = None) -> dict:
    """Age pre-current-quarter scratch notes into archive/<year>Q<n>/. Idempotent."""
    kb = Path(kb_dir or S.kb_dir())
    scratch, archive = kb / "scratch", kb / "archive"
    today = today or datetime.now(timezone.utc).date()
    cq_start = _quarter_start(today)
    moved: list[str] = []
    if scratch.exists():
        for datedir in sorted(p for p in scratch.iterdir() if p.is_dir()):
            try:
                ndate = date.fromisoformat(datedir.name)          # scratch/<iso-date>/
            except ValueError:
                continue                                          # not a date dir — leave it
            if ndate >= cq_start:
                continue                                          # current quarter stays in scratch
            qdest = archive / f"{ndate.year}Q{_quarter(ndate)}" / datedir.name
            for f in sorted(datedir.rglob("*")):
                if f.is_file():
                    target = qdest / f.relative_to(datedir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(target))
                    moved.append(str(target.relative_to(kb)))
            shutil.rmtree(datedir, ignore_errors=True)            # prune the emptied date dir
    return {"op": "upkeep", "moved": len(moved), "files": moved}


def snapshot(kb_dir: str | Path | None = None, now: datetime | None = None) -> dict:
    """Snapshot the projected current/ into archive/<year>Q<n>/_snapshot_<utc>/."""
    kb = Path(kb_dir or S.kb_dir())
    cur = kb / "current"
    now = now or datetime.now(timezone.utc)
    qdest = (kb / "archive" / f"{now.year}Q{_quarter(now.date())}"
             / f"_snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}")
    if cur.exists():
        shutil.copytree(cur, qdest)
    return {"op": "snapshot", "snapshot": str(qdest.relative_to(kb))}


def reproject() -> dict:
    """Re-run the projection so current/ tracks the catalog + on-disk runs."""
    from aegir.lineup import build
    build.run()
    return {"op": "reproject"}


# task_type → handler (payload dict → result dict). The processor dispatches on these.
HANDLERS = {
    "kb_archive": lambda payload: upkeep(),
    "kb_snapshot": lambda payload: snapshot(),
    "kb_reproject": lambda payload: reproject(),
}


def run(op: str, **kw) -> dict:
    if op == "upkeep":
        return upkeep(**kw)
    if op == "snapshot":
        return snapshot(**kw)
    if op == "reproject":
        return reproject()
    raise SystemExit(f"unknown maintain op {op!r} (upkeep|snapshot|reproject)")
