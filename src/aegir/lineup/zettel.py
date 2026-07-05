"""Run-zettels — idempotent run targets as an immutable, chained note system (RH 2026-07-05).

Lightly gaius-inspired zettelkasten discipline: ONE completed `just metaflow` run = ONE zettel,
permanently identified (``run-YYYYMMDD-HHMMSS``), never rewritten, linking backward to its
predecessor. Each zettel records exactly what made the run an idempotent target:

- the INPUT WINDOW — harvest cursor interval + passage counts (cached vs fresh)
- the VERSIONS — deriver_version stamp + repo commit
- the DELTAS — corpus counts, structure EMD, congruence before this zettel vs after

The chain lives in the corpus dir (``runs/*.json``) — pipeline truth, git-independent — and the
lineup projects it as ``corpus/runs/*`` notes so the KB carries the full accretion history.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_run_zettel(corpus_dir: Path, *, run_id: str, derive_stats: dict,
                     metrics: dict) -> Path:
    """Append one immutable run-zettel to ``corpus_dir/runs/``. Returns the zettel path."""
    corpus_dir = Path(corpus_dir)
    runs = corpus_dir / "runs"
    runs.mkdir(exist_ok=True)
    prior = sorted(runs.glob("run-*.json"))
    prev_id = prior[-1].stem if prior else None
    now = datetime.now(timezone.utc)
    zid = f"run-{now.strftime('%Y%m%d-%H%M%S')}"

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parents[3])).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    try:
        from aegir.ontology.derive_harness import deriver_version
        dv = deriver_version()
    except Exception:  # noqa: BLE001
        dv = ""
    cursor = {}
    cur_p = Path(__file__).resolve().parents[3] / "build/domain_harvest/cursor.json"
    if cur_p.exists():
        try:
            cursor = json.loads(cur_p.read_text())
        except Exception:  # noqa: BLE001
            pass

    zettel = {
        "id": zid,
        "prev": prev_id,
        "at": now.isoformat(timespec="seconds"),
        "metaflow_run_id": run_id,
        "window": {"harvest_cursor": cursor.get("cursor"),
                   "passages_cached": derive_stats.get("cached", 0),
                   "passages_fresh": sum(v for k, v in derive_stats.items()
                                         if k in ("rich", "thin", "inert", "malformed"))},
        "versions": {"deriver": dv, "commit": commit},
        "state": {
            "n_chapters": metrics.get("n_chapters"),
            "structure": metrics.get("structure"),
            "key_shapes": metrics.get("key_shapes"),
            "congruence": metrics.get("congruence"),
            "sensitive_ok": metrics.get("sensitive_ok"),
        },
    }
    out = runs / f"{zid}.json"
    out.write_text(json.dumps(zettel, indent=1, default=str))
    return out


def run_zettels(corpus_dir: Path) -> "list[dict]":
    """The chain, oldest→newest (graceful empty)."""
    runs = Path(corpus_dir) / "runs"
    out = []
    for p in sorted(runs.glob("run-*.json")) if runs.exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return out
