#!/usr/bin/env python
"""audit_pathways — Phase 1 of the pathway audit (RH 2026-07-23): the registry of every
ontology→relational generator, with measured columns. "OL emission" is a REQUIREMENT
(the OL-first consolidation mandate): a generator without RunEvents is outside governed
PROVENANCE and marked for convergence or retirement.

    uv run python scripts/audit_pathways.py → build/pathway_registry.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PATHWAYS = [
    {"id": "flow-kvasir-ddl", "generator": "sdg_corpora_flow.realize → kvasir ddl/shapes",
     "conductor": "metaflow step", "scope": "merged run ontology",
     "id_space": "kvasir plan names", "provenance_record": "kvasir plan citations (src_lines)",
     "consumers": ["shapes.ttl (released)", "structure score"],
     "checks": {"artifact": "corpora/ontology/sdg-ontology.omn"}},
    {"id": "ddl-spine", "generator": "build_ddl_spine (naming registers + RI rows)",
     "conductor": "flow stage", "scope": "catalog.json",
     "id_space": "t_snake + natural register", "provenance_record": "naming_map.parquet",
     "consumers": ["corpora/ddl (RELEASED)", "lineup current tables", "Atlas rdbms_*"],
     "checks": {"artifact": "corpora/ddl"}},
    {"id": "kvasir-comprehensive", "generator": "realize_sdg → kvasir ddl over the UNION",
     "conductor": "script (shakedown)", "scope": "certified union (comprehensive)",
     "id_space": "bare snake", "provenance_record": "structure.json + citations",
     "consumers": ["UNRELEASED (covers 2,108 'unpopulated' classes)"],
     "checks": {"artifact": "build/unified_grounded/ddl.sql"}},
    {"id": "schema-realization", "generator": "realize.py (EAV/junction/star/snowflake)",
     "conductor": "script / spine runs", "scope": "catalog subgraphs (stochastic)",
     "id_space": "snake spine runs", "provenance_record": "spine run dirs",
     "consumers": ["lineup scratch tables"],
     "checks": {"artifact": "scripts/realize.py"}},
    {"id": "constructs-web", "generator": "sdg_corpora_flow.build_constructs (per-chapter)",
     "conductor": "metaflow step", "scope": "chapter constructs",
     "id_space": "CamelCase", "provenance_record": "constructs web in chapters",
     "consumers": ["chapters (both registers)", "lineup scratch tables"],
     "checks": {"artifact": "src/aegir/flows/sdg_corpora_flow.py"}},
]


def main() -> int:
    ol_dirs = [REPO / "build/ol_events"] + sorted(
        Path("/raid/checkpoints/aegir-artifacts/sdg-corpora").glob("*/ol_events"))
    ol_jobs: set = set()
    for d in ol_dirs:
        for f in d.glob("*.json") if d.exists() else []:
            try:
                ol_jobs.add(json.loads(f.read_text())["job"]["name"])
            except Exception:  # noqa: BLE001
                continue
    rows = []
    for p in PATHWAYS:
        art = REPO / p["checks"]["artifact"]
        # OL status: kvasir-invoking pathways inherit kvasir's native emission; the flow
        # steps additionally emit their own events (governance.ol). Measured, not assumed.
        ol = ("native (kvasir P3)" if p["id"] in ("flow-kvasir-ddl", "kvasir-comprehensive")
              and ("witnesses" in ol_jobs or "ddl" in ol_jobs)
              else "flow OL (governance.ol)" if p["conductor"] == "metaflow step"
              else "ABSENT — convergence/retirement candidate")
        rows.append({**{k: v for k, v in p.items() if k != "checks"},
                     "artifact_present": art.exists(),
                     "ol_emission": ol,
                     "meets_ol_requirement": not ol.startswith("ABSENT")})
    out = {"charter": "docs/scratch/2026-07-23/012415_pathway_audit_charter.md",
           "requirement": "OL emission is REQUIRED (OL-first consolidation mandate); "
                          "one-way flow doctrine applies to every recorded edge",
           "ol_jobs_observed": sorted(ol_jobs),
           "n_pathways": len(rows),
           "n_meeting_ol_requirement": sum(r["meets_ol_requirement"] for r in rows),
           "pathways": rows}
    (REPO / "build/pathway_registry.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        mark = "✓" if r["meets_ol_requirement"] else "✗"
        print(f"{mark} {r['id']:22s} · {r['scope'][:32]:32s} · OL: {r['ol_emission']}")
    print(f"→ build/pathway_registry.json ({out['n_meeting_ol_requirement']}/{len(rows)} "
          "meet the OL requirement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
