"""The clearinghouse — strategy-independent judges for cross-strategy comparison (#148 inc-4).

Doctrine (locked in the design session): TARGETS live inside strategies, so the judges that
COMPARE strategies must be pinned OUTSIDE all of them, at the EXPERIMENT level — else a shadow
grades on its own curve. This module + ``judges.lock.json`` is that pin.

- The lock names each judge, its kind, and the artifacts it is pinned to (norms BY CONTENT
  HASH — a strategy that ships different targets is still scored on the common yardstick).
- ``evaluate(main_dir, shadow_dir)`` runs the mechanical judges over two corpus dirs and
  writes a symmetric report. Overwatch ADMINISTERS this (lifecycle, invocation, aggregation,
  escalation); it never scores content itself.
- Judge liveness = exit code + completion footer (PROJECT_001): a judge that dies mid-run
  must read as DEAD, not as a pass — every judge returns an explicit ``ok`` flag and the
  report carries ``judges_completed`` vs ``judges_declared``.
- CROSS-congruence: each arm's chapters scored against the OTHER arm's vocab collection
  (symmetrized) — reflexive congruence alone is self-grading and is never a comparison basis.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCK = Path(__file__).with_name("judges.lock.json")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def write_lock() -> dict:
    """Pin the current judge set. Run deliberately (an EXPERIMENT-level act, versioned in
    git) — never from inside a flow."""
    lock = {
        "version": 1,
        "doctrine": ["judges pinned outside all strategies",
                     "liveness = ok-flag + completed/declared counts (PROJECT_001)",
                     "cross-congruence symmetrized; reflexive congruence never compares"],
        "judges": {
            "structure_emd": {
                "kind": "fixed_reference",
                "norms": {"schemapile_shape_norms.json": _sha(REPO / "build/schemapile_shape_norms.json"),
                          "schemapile_key_norms.json": _sha(REPO / "build/schemapile_key_norms.json")},
                "note": "arms must have been scored against THESE norms (hash-checked); "
                        "a mismatch flags the arm, never silently rescales"},
            "naming_norms": {
                "kind": "mechanical",
                "measures": ["table_style distribution vs lock norms",
                             "leading-token (prefix) concentration",
                             "pk kind distribution"]},
            "cross_congruence": {
                "kind": "cross",
                "spec": "score each arm's chapters against the OTHER arm's vocab collection; "
                        "report the 2x2 (self is CONTEXT, cross is the comparison)"},
            "rubric": {
                "kind": "qualitative_mechanized",
                "entries": {
                    "stem_echo": r"\| ([a-z]{4,6})-(\d{3}) \|",
                    "adjacent_dup_cells": r"\| (\S{4,}) \| \1 \|",
                    "bare_id_alongside_surrogate": r"\|[^|\n]*_id \| id \|",
                    "arithmetic_dates": "diff-test on date columns (parser, not regex)",
                    "scaffold_monoculture": "block-sequence fingerprint (instrument PENDING — "
                                            "heading grep measured insufficient 2026-07-05)"}},
            "downstream_eval": {
                "kind": "deferred",
                "note": "the real arbiter (comprehension meta-objective); wire when #126 lands"},
        },
    }
    LOCK.write_text(json.dumps(lock, indent=1))
    return lock


def _corpus_state(d: Path) -> dict:
    m = {}
    for f in ("metrics.json", "ontology/structure.json"):
        p = d / f
        if p.exists():
            try:
                m[f] = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                pass
    return m


def _rubric_battery(d: Path, entries: dict, sample: int = 40) -> dict:
    import os
    files = sorted((d / "chapters").rglob("*.md"), key=os.path.getmtime)[-sample:]
    pats = {}
    for k, v in entries.items():
        if not isinstance(v, str):
            continue
        try:
            pats[k] = re.compile(v)
        except re.error:  # prose specs ("instrument PENDING…") are declarations, not regexes
            continue
    counts = {k: 0 for k in pats}
    for f in files:
        t = f.read_text(errors="replace")
        for k, rx in pats.items():
            counts[k] += len(rx.findall(t))
    return {"sampled": len(files), "counts": counts}


def evaluate(main_dir: Path, shadow_dir: Path, *, main_ref: str = "",
             shadow_ref: str = "") -> dict:
    """The mechanical comparison. Cross-congruence requires both arms' collections live —
    skipped (reported, not silently) when unavailable."""
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else write_lock()
    declared = list(lock["judges"])
    completed, report = [], {"arms": {"main": str(main_dir), "shadow": str(shadow_dir)},
                             "refs": {"main": main_ref, "shadow": shadow_ref}}

    a, b = _corpus_state(Path(main_dir)), _corpus_state(Path(shadow_dir))
    try:
        want = lock["judges"]["structure_emd"]["norms"]
        live = {k: _sha(REPO / "build" / k) for k in want}
        report["structure_emd"] = {
            "norms_match_lock": live == want,
            "main": (a.get("ontology/structure.json") or {}).get("shape_emd"),
            "shadow": (b.get("ontology/structure.json") or {}).get("shape_emd"),
            "ok": True}
        completed.append("structure_emd")
    except Exception as e:  # noqa: BLE001
        report["structure_emd"] = {"ok": False, "error": str(e)[:120]}

    try:
        ka = (a.get("metrics.json") or {}).get("key_shapes") or {}
        kb = (b.get("metrics.json") or {}).get("key_shapes") or {}
        report["naming_norms"] = {"main": ka, "shadow": kb, "ok": bool(ka or kb)}
        completed.append("naming_norms")
    except Exception as e:  # noqa: BLE001
        report["naming_norms"] = {"ok": False, "error": str(e)[:120]}

    try:
        entries = lock["judges"]["rubric"]["entries"]
        report["rubric"] = {"main": _rubric_battery(Path(main_dir), entries),
                            "shadow": _rubric_battery(Path(shadow_dir), entries), "ok": True}
        completed.append("rubric")
    except Exception as e:  # noqa: BLE001
        report["rubric"] = {"ok": False, "error": str(e)[:120]}

    try:
        from aegir.ontology.congruence import cross_congruence
        report["cross_congruence"] = cross_congruence(Path(main_dir), Path(shadow_dir),
                                                      main_ref=main_ref, shadow_ref=shadow_ref)
        report["cross_congruence"]["ok"] = True
        completed.append("cross_congruence")
    except Exception as e:  # noqa: BLE001 — collections down/unbuilt: reported, never silent
        report["cross_congruence"] = {"ok": False, "skipped": str(e)[:160]}

    report["judges_declared"] = declared
    report["judges_completed"] = completed
    report["ok"] = len(completed) >= 3
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "lock":
        lk = write_lock()
        print(f"locked {len(lk['judges'])} judges → {LOCK}")
    elif len(sys.argv) == 3:
        rep = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
        print(json.dumps(rep, indent=1, default=str)[:4000])
    else:
        print("usage: judges.py lock | judges.py <main_dir> <shadow_dir>")
