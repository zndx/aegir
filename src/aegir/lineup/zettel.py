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
    strategy = {}
    try:
        from aegir.strategy import lineage as _lin
        from aegir.strategy.manifest import declared, submodule_commit
        man = declared()
        if man:
            strategy = {"strategy_id": man["strategy_id"],
                        "strategy_commit": submodule_commit(),
                        "stage_keys": {st: _lin.stage_key(st, man)
                                       for st in _lin.STAGE_INPUTS}}
    except Exception:  # noqa: BLE001
        pass
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
        "versions": {"deriver": dv, "commit": commit, **strategy},
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


def chain_roots(zettels: "list[dict]") -> "dict[str, str]":
    """#147 zettel-lifecycle citizenship, computed from release CUT POINTS — nothing moves,
    the ROOT is computed at projection time.

    A release zettel (``kind == "release"`` — cut when a corpus is staged/released to
    sdg-corpora) closes the segment before it: the LATEST release's segment projects to
    ``current``, earlier releases' segments to ``archive``, and everything after the last
    cut — the unreleased accretion — to ``scratch``. With no release cut in the chain yet
    (today), the whole chain is unreleased → ``scratch``.
    """
    cuts = [i for i, z in enumerate(zettels) if z.get("kind") == "release"]
    last = cuts[-1] if cuts else -1
    prior = cuts[-2] if len(cuts) >= 2 else -1
    roots: dict[str, str] = {}
    for i, z in enumerate(zettels):
        if i > last:
            roots[z["id"]] = "scratch"
        elif i > prior:
            roots[z["id"]] = "current"
        else:
            roots[z["id"]] = "archive"
    return roots
