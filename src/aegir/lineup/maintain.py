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

import json
import re
import shutil
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

from aegir.lineup import notes as N
from aegir.lineup import sources as S

_WL = re.compile(r"\[\[([^\]|]+)((?:\|[^\]]*)?)\]\]")


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


def snapshot(kb_dir: str | Path | None = None, key: str | None = None,
             now: datetime | None = None) -> dict:
    """Freeze the projected ``current/`` into a NAMESPACED, self-contained archive snapshot
    under ``archive/<key>/`` — a point-in-time zettelkasten that coexists with a regenerated
    ``current`` and survives ``kb-build`` (which only rebuilds ``current``).

    Every note's id (frontmatter ``name``) and every ``[[wikilink]]`` is prefixed with
    ``<key>/`` so the snapshot is internally navigable and CANNOT collide with or bleed into
    the live (regen-latest) projection — clicking inside the snapshot stays in the snapshot.
    Writes a registry entry note (``archive/<key>.md``, kind ``archive-snapshot`` — the
    dropdown's entry point) and a provenance ``_manifest.json`` (the corpus/coverage/catalog
    it was projected from, so the snapshot is reproducible). ``key`` defaults to the calendar
    quarter; pass e.g. ``2026Q3`` to override."""
    kb = Path(kb_dir or S.kb_dir())
    cur = kb / "current"
    now = now or datetime.now(timezone.utc)
    key = key or f"{now.year}Q{_quarter(now.date())}"
    dest = kb / "archive" / key
    if dest.exists():
        shutil.rmtree(dest)
    if not cur.exists():
        return {"op": "snapshot", "key": key, "notes": 0, "note": "no current/ to snapshot"}

    n = 0
    for p in sorted(cur.rglob("*.md")):
        meta, body = N.parse_markdown(p.read_text())
        meta["name"] = f"{key}/{meta.get('name', p.relative_to(cur).as_posix()[:-3])}"
        meta["root"] = "archive"
        meta["links"] = [f"{key}/{x}" for x in (meta.get("links") or N.extract_links(body))]
        body = _WL.sub(lambda m: f"[[{key}/{m.group(1)}{m.group(2)}]]", body)
        head = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
        out = dest / p.relative_to(cur)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"---\n{head}\n---\n\n{body.rstrip()}\n")
        n += 1

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(S.REPO),
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        sha = "unknown"
    manifest = {"key": key, "taken_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "notes": n,
                "corpus_run": str(S.corpus_run() or ""), "coverage_run": str(S.coverage_run() or ""),
                "catalog_commit": sha, "source": "namespaced snapshot of current/ (pre-regen)"}
    (dest / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Registry entry — the dropdown's archive entry point (links into the namespaced snapshot).
    reg = N.Note(
        id=key, title=f"{key} — lineup snapshot", kind="archive-snapshot", data_product="content",
        root="archive",
        body=(f"**Frozen snapshot** of the `current` lineup projection — **{n} notes**, taken "
              f"{manifest['taken_utc']}.\n\n"
              f"Provenance: corpus `{Path(manifest['corpus_run']).parent.name or '—'}` · "
              f"catalog `{sha[:8]}`.\n\n"
              f"Browse: {N.wl(f'{key}/lens/terms', 'the collections × lens pivot')} · "
              f"{N.wl(f'{key}/collection/index', 'collections')} · "
              f"{N.wl(f'{key}/topic/index', 'topics')}\n"))
    N.write_note(kb, reg)
    # Refresh the index so the snapshot is immediately live (the registry entry + namespaced notes).
    entries = (N.scan_notes(kb, "current") + N.scan_notes(kb, "scratch") + N.scan_notes(kb, "archive"))
    N.write_index(kb, entries)
    return {"op": "snapshot", "key": key, "notes": n,
            "manifest": str((dest / "_manifest.json").relative_to(kb))}


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
