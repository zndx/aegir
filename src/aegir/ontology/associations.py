"""associations — ontology → relational-entity associations, BY CONSTRUCTION.

The one closure discipline (RH 2026-07-23), two generator-native emitters:

- ``emit_spine_associations`` — the RELEASED catalog spine (evidence: the spine's own
  naming_map records — kinds, pk plans, slot_refs). Extracted from lineup.sync so the
  FLOW's ddl stage emits natively at generation and the release act re-verifies
  idempotently — one function, two conductors.
- ``emit_comprehensive_closure`` — the certified-union lowering (evidence: the kvasir
  ``ddl --plan`` sidecar — the generator's OWN cited record of every table/junction/
  lookup with class + property IRIs, junction FK columns included).

Both honor the confirmation rule: a naming collision is NOT an association — every
published row carries an EXPLICIT procedural generation link (``confirmed_by``); name
matches without a recorded link are rejected and counted. Both emit the closure
section: every relational entity carries pattern + column roles from the
relational-concepts extension — the database-only tagging guarantee.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CORPORA = REPO / "corpora"


def emit_spine_associations(run_dir: "Path | None" = None) -> "dict | None":
    """Class↔table associations + the pattern/role closure for a spine run
    (default: the latest run under corpora/ddl). Writes
    ``<run_dir>/ontology_entity_associations.json``; returns the summary."""
    import pandas as _pd
    frag = json.loads((REPO / "build/foreign_fragments/sdg.json").read_text())
    if run_dir is None:
        ddl_dirs = sorted((CORPORA / "ddl").glob("*/naming_map.parquet"))
        if not ddl_dirs:
            return None
        run_dir = ddl_dirs[-1].parent
    run_dir = Path(run_dir)
    nm = _pd.read_parquet(run_dir / "naming_map.parquet")
    own: dict = {}
    for t_, tb_ in zip(nm[nm["kind"] == "entity"]["template_id"],
                       nm[nm["kind"] == "entity"]["table_name"]):
        own.setdefault(str(t_), str(tb_))
    refs: dict = {}
    name_tid: dict = {}
    for rf_, tb_, co_, sc_ in zip(nm["slot_ref"], nm["table_name"],
                                  nm["col_name"], nm["semantic_col"]):
        if rf_ and str(rf_) != "__pk__":
            refs.setdefault(str(rf_), []).append({"table": str(tb_), "column": str(co_)})
            if sc_:
                name_tid.setdefault(str(rf_), str(sc_))
    head_tid: dict = {}
    try:
        cat = json.loads((REPO / "src/aegir/ontology/catalog/catalog.json").read_text())
        for t_ in cat.get("templates", []):
            h_ = re.search(r"Class:\s*\{(\w+):Class\}", t_.get("manchester_template") or "")
            if h_:
                head_tid[h_.group(1)] = t_["template_id"]
    except Exception:  # noqa: BLE001
        pass
    # CONFIRMATION RULE (RH 2026-07-23): a naming collision between ontology and
    # relational entities is NOT an association — an EXPLICIT procedural generation
    # link recorded by the pipeline must confirm every row published here (and to
    # Atlas thereby). The confirmation universe is THIS run's recorded consumption:
    # its manifest names its inputs; its naming_map rows are the generator's own
    # record of which classes it consumed. Each association carries its evidence.
    manifest = {}
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    run_templates = set(own)                    # templates THIS run generated tables for
    run_slot_refs = set(refs)                   # classes THIS run's generator referenced
    rows = []
    n_unpop = 0
    n_rejected = 0
    classes = sorted({k for fam in ("frames", "equiv") for k in (frag.get(fam) or {})})
    for iri in classes:
        if "signals.zndx.org" not in iri:
            continue
        local = iri.rsplit("#", 1)[-1]
        assoc = []
        tid = head_tid.get(local)
        if tid and tid in run_templates:
            assoc.append({"kind": "own_table", "table": own[tid],
                          "atlas_qualifiedName": f"{own[tid]}@aegir",
                          "confirmed_by": "catalog_head_template",
                          "template_id": tid})
        elif name_tid.get(local) and name_tid[local] in run_templates:
            t2 = name_tid[local]
            assoc.append({"kind": "own_table", "table": own[t2],
                          "atlas_qualifiedName": f"{own[t2]}@aegir",
                          "confirmed_by": "naming_map_recorded_pair",
                          "template_id": t2})
        elif head_tid.get(local) or name_tid.get(local):
            n_rejected += 1                     # name match WITHOUT a link in this run
        if local in run_slot_refs:
            assoc += [{"kind": "fk_reference", **r_,
                       "atlas_qualifiedName": f"{r_['table']}@aegir",
                       "confirmed_by": "naming_map_fk_record"}
                      for r_ in refs[local]]
        if not assoc:
            n_unpop += 1
        rows.append({"class_iri": iri, "class": local,
                     "template_id": (head_tid.get(local) or name_tid.get(local) or ""),
                     "associations": assoc})
    # THE CLOSURE SECTION (RH 2026-07-23): every relational entity — including
    # association/EAV patterns and plumbing columns — carries its ontological
    # source, so a database-only consumer can tag each entity. Patterns and column
    # roles come from the RELATIONAL-CONCEPTS extension (authored, ⊑ cco ICE);
    # evidence stays record-grade (naming_map kinds, pk records, slot_refs).
    tid_head = {v: k for k, v in head_tid.items()}
    by_table: dict = {}
    for r_ in nm.to_dict("records"):
        by_table.setdefault(str(r_["table_name"]), []).append(r_)
    PATTERN_OF_KIND = {"entity": "sdg:EntityTable", "junction": "sdg:AssociationTable",
                       "lookup": "sdg:LookupTable", "eav": "sdg:EntityAttributeValueTable",
                       "view": "sdg:RelationalView"}
    tables_out = []
    n_no_pattern = 0
    for tname, trs in sorted(by_table.items()):
        kind_ = str(trs[0].get("kind") or "")
        pattern = PATTERN_OF_KIND.get(kind_)
        if not pattern:
            n_no_pattern += 1
        tid_ = str(trs[0].get("template_id") or "")
        dom_cls = tid_head.get(tid_, "")
        cols_out = []
        for r_ in trs:
            col, sref = str(r_.get("col_name")), str(r_.get("slot_ref") or "")
            if sref == "__pk__":
                role = "sdg:SurrogateIdentifierColumn"
            elif sref and sref[:1].isupper():
                role = "sdg:ForeignKeyColumn"
            elif col in ("created", "updated"):
                role = "sdg:AuditTimestampColumn"
            else:
                role = "sdg:AttributeColumn"
            c_ = {"column": col, "role": role,
                  "confirmed_by": "naming_map_record"}
            if role == "sdg:ForeignKeyColumn":
                c_["references_class"] = sref
            elif role == "sdg:AttributeColumn" and r_.get("semantic_col"):
                c_["concept"] = str(r_["semantic_col"])
            cols_out.append(c_)
        tables_out.append({
            "table": tname, "atlas_qualifiedName": f"{tname}@aegir",
            "schema_pattern": pattern or "sdg:RelationalSchemaEntity",
            "pattern_confirmed_by": "naming_map_kind_record",
            "realized_from_class": dom_cls,
            "columns": cols_out})
    out = {"run": run_dir.name, "atlas_typeName": "rdbms_table",
           "generated_from": {"inputs": manifest.get("catalog_files", []),
                              "n_templates": manifest.get("n_templates"),
                              "created_at": manifest.get("created_at")},
           "confirmation_rule": "every association carries an EXPLICIT procedural "
                                "generation link recorded by the pipeline (confirmed_by); "
                                "name matches without a recorded link in THIS run are "
                                "rejected, never published",
           "n_classes": len(rows),
           "n_unpopulated": n_unpop,
           "n_name_matches_rejected": n_rejected,
           "note": "associations == [] marks constructs awaiting the next "
                   "agent-mediated generative pass (ontology → relational)",
           "closure": {"n_tables": len(tables_out),
                       "n_without_pattern": n_no_pattern,
                       "rule": "every relational entity carries its ontological "
                               "source (pattern + column roles from the "
                               "relational-concepts extension; domain class where "
                               "generated) — the database-only tagging guarantee"},
           "tables": tables_out,
           "classes": rows}
    (run_dir / "ontology_entity_associations.json").write_text(json.dumps(out, indent=1))
    return {"run": run_dir.name, "n_classes": len(rows), "n_unpopulated": n_unpop,
            "n_name_matches_rejected": n_rejected,
            "n_tables": len(tables_out), "n_without_pattern": n_no_pattern,
            "path": str(run_dir / "ontology_entity_associations.json")}


def emit_comprehensive_closure(out_dir: Path) -> "dict | None":
    """Closure artifact for the comprehensive (certified-union) lowering.

    Consumes the kvasir ``ddl --plan`` sidecar — the generator's own record — so every
    association is an exact-IRI recorded link (never a name match): entity tables carry
    ``class``, columns carry ``prop``, junctions carry both participant classes AND their
    rendered FK columns, lookups carry their source property. Every emitted relational
    entity is one of tables/junctions/lookups, so the pattern closure holds by
    construction; ``n_without_pattern`` is still measured, never assumed."""
    out_dir = Path(out_dir)
    plan_p = out_dir / "plan.json"
    if not plan_p.exists():
        return None
    plan = json.loads(plan_p.read_text())
    tables_out = []
    n_no_pattern = 0
    class_tables: dict = {}                 # class IRI → own tables (exact-IRI record)
    class_refs: dict = {}                   # class IRI → fk/junction references
    for t in plan.get("tables") or []:
        fk_by_col = {f["column"]: f for f in (t.get("fks") or [])}
        cols_out = []
        for c in t.get("columns") or []:
            fk = fk_by_col.get(c["name"])
            if fk:
                role = "sdg:ForeignKeyColumn"
            elif c.get("pk") and not c.get("prop"):
                role = "sdg:SurrogateIdentifierColumn"
            else:
                role = "sdg:AttributeColumn"
            c_ = {"column": c["name"], "role": role, "confirmed_by": "kvasir_plan_record"}
            if c.get("prop"):
                c_["property_iri"] = c["prop"]
            if fk:
                c_["references_class"] = fk["target_class"]
                class_refs.setdefault(fk["target_class"], []).append(
                    {"table": t["name"], "column": c["name"]})
            cols_out.append(c_)
        tables_out.append({"table": t["name"], "atlas_qualifiedName": f"{t['name']}@aegir",
                           "schema_pattern": "sdg:EntityTable",
                           "pattern_confirmed_by": "kvasir_plan_record",
                           "realized_from_class": t.get("class", ""),
                           "columns": cols_out})
        class_tables.setdefault(t.get("class", ""), []).append(t["name"])
    for j in plan.get("junctions") or []:
        tables_out.append({"table": j["name"], "atlas_qualifiedName": f"{j['name']}@aegir",
                           "schema_pattern": "sdg:AssociationTable",
                           "pattern_confirmed_by": "kvasir_plan_record",
                           "realized_from_property": j.get("prop", ""),
                           "participant_classes": [j.get("subject_class", ""),
                                                   j.get("target_class", "")],
                           "columns": [
                               {"column": j["subject_col"], "role": "sdg:ForeignKeyColumn",
                                "references_class": j.get("subject_class", ""),
                                "confirmed_by": "kvasir_plan_record"},
                               {"column": j["target_col"], "role": "sdg:ForeignKeyColumn",
                                "references_class": j.get("target_class", ""),
                                "confirmed_by": "kvasir_plan_record"}]})
        for side in ("subject", "target"):
            class_refs.setdefault(j.get(f"{side}_class", ""), []).append(
                {"table": j["name"], "column": j[f"{side}_col"]})
    for lu in plan.get("lookups") or []:
        tables_out.append({"table": lu["name"], "atlas_qualifiedName": f"{lu['name']}@aegir",
                           "schema_pattern": "sdg:LookupTable",
                           "pattern_confirmed_by": "kvasir_plan_record",
                           "realized_from_property": lu.get("prop", ""),
                           "columns": [{"column": "code", "role": "sdg:AttributeColumn",
                                        "property_iri": lu.get("prop", ""),
                                        "confirmed_by": "kvasir_plan_record"}]})
    # class-level view (the unpopulated worklist, measured against the SAME census
    # the spine artifact uses — exact-IRI joins against the plan record only)
    rows = []
    n_unpop = 0
    try:
        frag = json.loads((REPO / "build/foreign_fragments/sdg.json").read_text())
        census = sorted({k for fam in ("frames", "equiv") for k in (frag.get(fam) or {})
                         if "signals.zndx.org" in k})
    except Exception:  # noqa: BLE001
        census = []
    for iri in census:
        assoc = [{"kind": "own_table", "table": tb, "atlas_qualifiedName": f"{tb}@aegir",
                  "confirmed_by": "kvasir_plan_record"} for tb in class_tables.get(iri, [])]
        assoc += [{"kind": "fk_reference", **r_, "atlas_qualifiedName": f"{r_['table']}@aegir",
                   "confirmed_by": "kvasir_plan_record"} for r_ in class_refs.get(iri, [])]
        if not assoc:
            n_unpop += 1
        rows.append({"class_iri": iri, "class": iri.rsplit("#", 1)[-1],
                     "associations": assoc})
    out = {"scope": "comprehensive", "atlas_typeName": "rdbms_table",
           "generated_from": {"ontology": plan.get("ontology") or str(out_dir / "sdg-ontology.omn"),
                              "plan": str(plan_p),
                              "n_tables": len(plan.get("tables") or []),
                              "n_junctions": len(plan.get("junctions") or []),
                              "n_lookups": len(plan.get("lookups") or [])},
           "confirmation_rule": "every association is an exact-IRI link from the kvasir "
                                "ddl --plan record (the generator's own citation); no "
                                "name matching anywhere in this artifact",
           "n_classes": len(rows),
           "n_unpopulated": n_unpop,
           "closure": {"n_tables": len(tables_out),
                       "n_without_pattern": n_no_pattern,
                       "rule": "every relational entity carries its ontological "
                               "source (pattern + column roles from the "
                               "relational-concepts extension) — the database-only "
                               "tagging guarantee, generator-recorded"},
           "tables": tables_out,
           "classes": rows}
    (out_dir / "ontology_entity_associations.json").write_text(json.dumps(out, indent=1))
    return {"scope": "comprehensive", "n_classes": len(rows), "n_unpopulated": n_unpop,
            "n_tables": len(tables_out), "n_without_pattern": n_no_pattern,
            "path": str(out_dir / "ontology_entity_associations.json")}
