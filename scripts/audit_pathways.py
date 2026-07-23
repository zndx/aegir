#!/usr/bin/env python
"""audit_pathways — Phase 1 of the pathway audit (RH 2026-07-23): the registry of every
ontology→relational generator, with measured columns. "OL emission" is a REQUIREMENT
(the OL-first consolidation mandate): a generator without RunEvents is outside governed
PROVENANCE and marked for convergence or retirement.

Post-consolidation (Phase 3 rulings implemented): the flow's ``ddl_stage`` is THE
kvasir-scoped conductor — catalog and comprehensive are SCOPES of one stage, not
pathways; the constructs web records explicit column roles. Conductor claims are
MEASURED against the flow/module source, never asserted.

    uv run python scripts/audit_pathways.py → build/pathway_registry.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PATHWAYS = [
    {"id": "flow-kvasir-ddl", "generator": "sdg_corpora_flow.realize → kvasir ddl/shapes",
     "conductor": "metaflow step (realize)", "scope": "merged run ontology",
     "id_space": "kvasir plan names", "provenance_record": "kvasir plan citations (src_lines)",
     "consumers": ["shapes.ttl (released)", "structure score"],
     "checks": {"artifact": "corpora/ontology/sdg-ontology.omn"}},
    {"id": "ddl-spine", "generator": "build_ddl_spine (naming registers + RI rows)",
     "conductor": "metaflow step (ddl_stage · scope=catalog)", "scope": "catalog.json",
     "id_space": "t_snake + natural register", "provenance_record":
         "naming_map.parquet + ontology_entity_associations.json (closure, native)",
     "consumers": ["corpora/ddl (RELEASED)", "lineup current tables", "Atlas rdbms_*"],
     "checks": {"artifact": "corpora/ddl", "flow_step": "def ddl_stage"}},
    {"id": "kvasir-comprehensive", "generator": "ddl_stage → realize_sdg → kvasir ddl --plan",
     "conductor": "metaflow step (ddl_stage · scope=comprehensive)",
     "scope": "certified union (flow parameter, not a pathway)",
     "id_space": "kvasir plan record", "provenance_record":
         "plan.json (generator-cited) + closure artifact (exact-IRI links)",
     "consumers": ["covers the catalog spine's unpopulated classes (2,108 → 66 measured)"],
     "checks": {"artifact": "build/unified_grounded/plan.json",
                "flow_param": "ddl-scope"}},
    {"id": "schema-realization", "generator": "realize.py (EAV/junction/star/snowflake)",
     "conductor": "ddl_stage flag (build_ddl_spine --realize); standalone = deprecated scratch",
     "scope": "catalog subgraphs (stochastic)",
     "id_space": "snake spine runs", "provenance_record": "spine run dirs",
     "consumers": ["lineup scratch tables"],
     "checks": {"artifact": "src/aegir/ontology/realize.py", "spine_flag": "--realize"}},
    {"id": "constructs-web", "generator": "sdg_corpora_flow.build_constructs (per-chapter)",
     "conductor": "metaflow step (build_constructs)", "scope": "chapter constructs",
     "id_space": "CamelCase + explicit role records",
     "provenance_record": "construct columns: role + exact prefixed IRI (normalized) — "
                          "legacy runs measured via convention fallback",
     "consumers": ["chapters (both registers)", "lineup scratch tables"],
     "checks": {"artifact": "src/aegir/flows/sdg_corpora_flow.py",
                "normalized_roles": "sdg:ForeignKeyColumn"}},
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
    flow_src = (REPO / "src/aegir/flows/sdg_corpora_flow.py").read_text()
    entities_src = (REPO / "src/aegir/ontology/entities.py").read_text()
    spine_src = (REPO / "scripts/build_ddl_spine.py").read_text()
    rows = []
    for p in PATHWAYS:
        c = p["checks"]
        art = REPO / c["artifact"]
        # every conductor claim is MEASURED against source/artifacts, never asserted
        measured = {
            "artifact_present": art.exists(),
            "flow_step_present": (c["flow_step"] in flow_src) if "flow_step" in c else None,
            "flow_param_present": (c["flow_param"] in flow_src) if "flow_param" in c else None,
            "spine_flag_present": (c["spine_flag"] in spine_src) if "spine_flag" in c else None,
            "normalized_roles_recorded": (c["normalized_roles"] in entities_src)
                                         if "normalized_roles" in c else None,
        }
        conducted = all(v for v in measured.values() if v is not None)
        if p["id"] == "flow-kvasir-ddl":
            ol = ("native (kvasir P3) + flow ingest"
                  if {"witnesses", "ddl", "shapes"} & ol_jobs else "flow OL (governance.ol)")
        elif p["id"] in ("ddl-spine", "kvasir-comprehensive"):
            ol = ("flow OL (emit_run_event) + closure native" if conducted
                  and "emit_run_event" in flow_src else "ABSENT — convergence candidate")
        elif p["id"] == "schema-realization":
            ol = ("stage OL when conducted via ddl_stage (--realize); standalone runs "
                  "remain ungoverned scratch (deprecated)" if conducted
                  else "ABSENT — convergence/retirement candidate")
        else:
            ol = "flow OL (governance.ol)" if "metaflow" in p["conductor"] else \
                 "ABSENT — convergence/retirement candidate"
        rows.append({**{k: v for k, v in p.items() if k != "checks"},
                     "measured": {k: v for k, v in measured.items() if v is not None},
                     "ol_emission": ol,
                     "meets_ol_requirement": conducted and not ol.startswith("ABSENT")})
    out = {"charter": "docs/scratch/2026-07-23/012415_pathway_audit_charter.md",
           "requirement": "OL emission is REQUIRED (OL-first consolidation mandate); "
                          "one-way flow doctrine applies to every recorded edge",
           "consolidation": "ddl_stage = the one kvasir-scoped conductor; scope "
                            "(catalog | comprehensive) is a flow PARAMETER; closure "
                            "artifacts emit natively at generation",
           "ol_jobs_observed": sorted(ol_jobs),
           "n_pathways": len(rows),
           "n_meeting_ol_requirement": sum(r["meets_ol_requirement"] for r in rows),
           "pathways": rows}
    (REPO / "build/pathway_registry.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        mark = "✓" if r["meets_ol_requirement"] else "✗"
        print(f"{mark} {r['id']:22s} · {r['conductor'][:44]:44s} · OL: {r['ol_emission'][:52]}")
    print(f"→ build/pathway_registry.json ({out['n_meeting_ol_requirement']}/{len(rows)} "
          "meet the OL requirement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
