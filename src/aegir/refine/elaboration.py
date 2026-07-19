"""Escalation triage — the standing consumer of the elaboration ACP worklist (#32, RH-ratified).

Never-drop makes the worklist a PERMANENT ORGAN: every membrane run exhales escalations by design,
so the complete procedure must include their consumer or the doctrine silts up. This module is that
consumer, run as the ``refine_escalations`` step of ``SdgCorporaFlow`` (and standalone via
``python -m aegir.refine.elaboration``).

Contract (the ratified conditions):
  * success = PROCESSED, not emptied — the residual worklist persisting is a green outcome;
  * stratified: mechanical failures (placeholder mismatch) first, the hard tail after,
    engine-unavailable entries retry free (no attempt burned);
  * bounded: per-entry ACP rounds + a global entry budget per run;
  * aged: after ``AGING_K`` failed pipeline passes an entry moves to the ``review`` stratum
    (surfaced to humans; no more engine cycles);
  * disposed by the SAME membranes as the single-shot loop (parse → CLT floor → drift-routed
    direction-aware judge) — the agent proposes, membranes dispose;
  * provenanced: one OL run per triage (inputs: the worklist; outputs: the refined-template
    dataset), and KPIs (size / age histogram / cleared) returned for the run's metrics.json.

The proposer is the fork-venv ACP subprocess (``aegir.refine._propose`` mode="elaborations") —
the channel where the agent sees the FULL membrane dialogue, which is exactly what these entries
need: each already failed 3 single-shot rounds on terse feedback.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FORK_PY = ROOT / "components" / "oss-mistral-cli" / ".venv" / "bin" / "python"
WORKLIST = ROOT / "build" / "elaboration_acp_worklist.json"
CATALOG = ROOT / "src" / "aegir" / "ontology" / "catalog" / "catalog.json"

AGING_K = 3          # failed triage passes before an entry leaves the engine path for human review
MECHANICAL = ("placeholder set mismatch", "length out of bounds", "reads as reasoning")


def _membranes():
    """The single source of membrane truth is the v2 script — import it rather than re-state the
    gates (parse, CLT floor with calibrated τ, drift-routed direction-aware judge)."""
    spec = importlib.util.spec_from_file_location(
        "elab_v2", ROOT / "scripts" / "elaborate_verbalizations_v2.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stratum(entry: dict) -> str:
    reasons = str(entry.get("reasons") or "")
    if int(entry.get("attempts") or 0) >= AGING_K:
        return "review"
    if "judge unavailable" in reasons:
        return "retry"                       # engine hiccup — costs no attempt
    if any(k in reasons for k in MECHANICAL):
        return "mechanical"
    return "hard"


def _propose(entry: dict, n: int, timeout: int = 480) -> "tuple[list, dict | None]":
    req = json.dumps({"construct": {"base": entry.get("base", ""),
                                    "manchester": entry.get("manchester", ""),
                                    "dialogue": entry.get("dialogue") or [entry.get("reasons", "")],
                                    "n": n},
                      "mode": "elaborations"})
    p = subprocess.run([str(FORK_PY), "-m", "aegir.refine._propose"], input=req,
                       capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    if p.returncode != 0:
        return [], None
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        return [], None
    return out.get("elaborations") or [], out.get("_exchange")


def _emit_lineage(run_key: str, kpis: dict) -> dict:
    """One idempotent OL run for the triage (the Atlas+OL provenance-authority posture — refined
    templates are as auditable as harvested pools). Down graph = legitimate offline state."""
    try:
        from aegir.governance import ol
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ol.PRODUCER}/refine_escalations/{run_key}"))
        event = {"eventType": "COMPLETE",
                 "eventTime": datetime.now(timezone.utc).isoformat(),
                 "producer": ol.PRODUCER,
                 "run": {"runId": run_id},
                 "job": {"namespace": "aegir", "name": "refine_escalations"},
                 "inputs": [{"namespace": "aegir", "name": "elaboration_acp_worklist",
                             "facets": {"worklist": {"n_in": kpis["worklist_in"]}}}],
                 "outputs": [{"namespace": "aegir", "name": "catalog_verbal_templates",
                              "facets": {"triage": kpis}}]}
        return ol.ingest_run_event(event)
    except Exception:  # noqa: BLE001
        return {"lineage": "unavailable"}


def triage(*, budget: int = 24, n_per_entry: int = 3, run_key: str = "manual",
           worklist_path: Path = WORKLIST, catalog_path: Path = CATALOG) -> dict:
    """One bounded triage pass. Returns the KPI dict (also what the flow writes into metrics.json)."""
    if not worklist_path.exists():
        return {"worklist_in": 0, "processed": 0, "accepted": 0, "cleared": 0,
                "aged_to_review": 0, "residual": 0, "skipped": "no worklist"}
    worklist = json.loads(worklist_path.read_text())
    cat = json.loads(catalog_path.read_text())
    tpls = cat.get("templates", cat if isinstance(cat, list) else [])
    by_id = {t["template_id"]: t for t in tpls}

    strata: "dict[str, list]" = {"mechanical": [], "hard": [], "retry": [], "review": []}
    for key, e in worklist.items():
        strata[_stratum(e)].append((key, e))
    order = strata["retry"] + strata["mechanical"] + strata["hard"]

    v2 = _membranes()
    clt = v2.CltMembrane()
    kpis = {"worklist_in": len(worklist), "processed": 0, "accepted": 0, "cleared": 0,
            "aged_to_review": len(strata["review"]), "engine_down_held": 0}
    for key, entry in order[:budget]:
        tid = entry.get("template_id") or key.split("::")[0]
        t = by_id.get(tid)
        if t is None:                                    # template gone — worklist hygiene
            worklist.pop(key, None)
            continue
        entry.setdefault("manchester", t.get("manchester_template", ""))
        base = entry.get("base", "")
        cands, exchange = _propose(entry, n_per_entry)
        kpis["processed"] += 1
        if exchange is None:                             # transport/agent failure — no attempt burned
            kpis["engine_down_held"] += 1
            continue
        base_slots = v2.slots_of(base)
        accepted, reasons = [], []
        for cand in cands:
            ok, why = v2.m_parse(cand, base_slots)
            if not ok:
                reasons.append(why)
                continue
            verdict, j = clt.check(base, cand)
            if verdict == "fail":
                reasons.append(f"feature-space floor (J={j:.2f} < τ={clt.tau})")
                continue
            if verdict == "pass" and v2.subject_drift(base, cand):
                verdict = "margin"
            if verdict == "margin":
                ok, why = v2.m_judge(base, cand)
                if not ok:
                    reasons.append(why)
                    continue
            accepted.append(cand)
        if accepted:
            pool = t.get("verbal_templates") or ([t["verbal_template"]] if t.get("verbal_template") else [])
            merged = pool + [c for c in accepted if c not in pool]
            t["verbal_templates"] = merged
            t.setdefault("provenance", {})["refined"] = f"refine-escalations/{run_key}"
            kpis["accepted"] += len(accepted)
            kpis["cleared"] += 1
            worklist.pop(key, None)
        else:
            entry["attempts"] = int(entry.get("attempts") or 0) + 1
            dlg = entry.setdefault("dialogue", [entry.get("reasons", "")])
            dlg.append("; ".join(dict.fromkeys(reasons))[:400] or "no candidates produced")
            if entry["attempts"] >= AGING_K:
                entry["stratum"] = "review"
                kpis["aged_to_review"] += 1
            worklist[key] = entry

    kpis["residual"] = len(worklist)
    ages = {}
    for e in worklist.values():
        ages[str(e.get("attempts") or 0)] = ages.get(str(e.get("attempts") or 0), 0) + 1
    kpis["age_histogram"] = ages
    catalog_path.write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    worklist_path.write_text(json.dumps(worklist, indent=1))
    kpis["lineage"] = _emit_lineage(run_key, {k: v for k, v in kpis.items() if k != "lineage"})
    return kpis


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--run-key", default="manual")
    a = ap.parse_args()
    kpis = triage(budget=a.budget, n_per_entry=a.n, run_key=a.run_key)
    print(json.dumps(kpis, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
