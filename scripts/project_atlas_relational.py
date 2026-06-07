#!/usr/bin/env python3
"""Atlas relational projector (retrofit / paper-prototype).

Projects the ontology's RELATIONAL footprint + synthesized corpus VIEWS into Apache
Atlas v2, for N families. Design: docs/scratch/2026-06-07/020525_atlas_relational_projector.md.

Modality (tier-1 retrofit, ontology-sourced base):
  - Base tables  = ontology templates  -> hive_table in db `footprint`
  - Base columns = template slots       -> hive_column (type = slot OWL metatype)
  - Corpus views = synthesized projections/joins over the base -> hive_table in db `corpus`
  - Lineage      = Process(base tables -> view)
  - Classifications: CTA (per slot OWL type) on columns; domain (per family) + BFO anchor on tables; CPA (complex-axiom) on columns
  - Business metadata: OntologyProvenance (bfo_anchor / template_id / manchester / is_complex / family) on base tables
  - Glossary: one glossary `Aegir Ontology`, a category per family, a term per base table (assigned to the table)

Idempotent: Atlas entity create is upsert-by-qualifiedName; types/glossary are existence-checked.
Run:  uv run --no-sync python scripts/project_atlas_relational.py [--per-family 15] [--reset]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402

ATLAS = "http://127.0.0.1:21000"
CLUSTER = "aegir"
CATALOG_DIR = Path("src/aegir/ontology/catalog")
FAMILIES = ["01_foundation", "02_observation_measurement", "05_provo_lineage"]

S = requests.Session()
S.auth = ("admin", "admin")
S.headers["Content-Type"] = "application/json"


def api(method: str, path: str, timeout: int = 120, **kw):
    return S.request(method, f"{ATLAS}/api/atlas/v2/{path}", timeout=timeout, **kw)


def _post_chunk(ents):
    """POST one entity/bulk chunk (long timeout — AGE entity create is slow); return guidAssignments."""
    r = api("POST", "entity/bulk", json={"entities": ents}, timeout=600)
    if not r.ok:
        print("   chunk FAILED:", r.status_code, r.text[:300])
        return {}
    return r.json().get("guidAssignments", {})


def fam_label(f: str) -> str:
    return f.split("_", 1)[1]  # "01_foundation" -> "foundation"


# ── type system ──────────────────────────────────────────────────────────────

OWL_TYPES = ["Class", "ObjectProperty", "DataProperty", "Individual", "Datatype", "AnnotationProperty"]


def cls_attr(name, desc):
    return {"name": name, "typeName": "string", "isOptional": True, "cardinality": "SINGLE",
            "valuesMinCount": 0, "valuesMaxCount": 1, "isUnique": False, "isIndexable": True,
            "description": desc}


def ensure_types():
    existing = {h["name"] for h in api("GET", "types/typedefs/headers").json()}
    class_defs, bm_defs = [], []

    def cdef(name, desc):
        return {"name": name, "description": desc, "superTypes": [],
                "attributeDefs": [cls_attr("confidence", "annotation confidence 0..1"),
                                  cls_attr("evidence", "source IRI / template / slot")]}

    # CTA: column semantic type (per OWL metatype)
    for o in OWL_TYPES:
        n = f"cta_{o.lower()}"
        if n not in existing:
            class_defs.append(cdef(n, f"CTA: column whose ontology slot type is OWL {o}"))
    # domain: table family
    for f in FAMILIES:
        n = f"domain_{fam_label(f)}"
        if n not in existing:
            class_defs.append(cdef(n, f"Domain: table realized from ontology family {f}"))
    # CPA: column property
    for n, d in [("cpa_complex_axiom", "CPA: column from a complex (DeepOnto) axiom template"),
                 ("cpa_bfo_process", "CPA: table anchored under bfo:Process")]:
        if n not in existing:
            class_defs.append(cdef(n, d))

    # Business metadata: ontology provenance
    if "OntologyProvenance" not in existing:
        def bm_attr(name, tn="string"):
            return {"name": name, "typeName": tn, "isOptional": True, "cardinality": "SINGLE",
                    "valuesMinCount": 0, "valuesMaxCount": 1, "isUnique": False, "isIndexable": True,
                    "options": {"applicableEntityTypes": "[\"hive_table\",\"hive_column\"]",
                                "maxStrLength": "2000"}}
        bm_defs.append({"name": "OntologyProvenance",
                        "description": "Provenance from the grounding ontology",
                        "attributeDefs": [bm_attr("template_id"), bm_attr("family"),
                                          bm_attr("bfo_anchor"), bm_attr("manchester"),
                                          bm_attr("is_complex", "boolean")]})

    if class_defs or bm_defs:
        payload = {"classificationDefs": class_defs, "businessMetadataDefs": bm_defs,
                   "enumDefs": [], "structDefs": [], "entityDefs": [], "relationshipDefs": []}
        r = api("POST", "types/typedefs", json=payload)
        print(f"  types: +{len(class_defs)} classifications, +{len(bm_defs)} businessMetadata -> {r.status_code}")
        if not r.ok:
            print("   ", r.text[:400])
    else:
        print("  types: already present")


# ── glossary ─────────────────────────────────────────────────────────────────

def ensure_glossary(tables_by_fam):
    glos = api("GET", "glossary").json() or []
    g = next((x for x in glos if x.get("name") == "Aegir Ontology"), None)
    if g is None:
        g = api("POST", "glossary", json={"name": "Aegir Ontology",
                "shortDescription": "BFO/CCO-grounded ontology terms backing the relational footprint"}).json()
    gguid = g["guid"]
    # existing categories/terms by name (refetch detailed)
    detail = api("GET", f"glossary/{gguid}/detailed").json()
    cat_guid = {c["displayText"]: c["categoryGuid"] for c in detail.get("categories", [])}
    term_guid = {t["displayText"]: t["termGuid"] for t in detail.get("terms", [])}

    for f in FAMILIES:
        cname = fam_label(f)
        if cname not in cat_guid:
            r = api("POST", "glossary/category", json={"name": cname, "anchor": {"glossaryGuid": gguid}})
            if not r.ok:
                print(f"   category {cname}: {r.status_code} {r.text[:200]}"); continue
            cat_guid[cname] = r.json().get("guid")

    # one term per base table (template), assigned to the table
    made = 0
    for f, tables in tables_by_fam.items():
        cname = fam_label(f)
        for t in tables:
            tname = t["template_id"]
            if tname in term_guid:
                continue
            body = {"name": tname, "anchor": {"glossaryGuid": gguid},
                    "shortDescription": t["manchester"][:240],
                    "categories": [{"categoryGuid": cat_guid[cname]}]}
            r = api("POST", "glossary/term", json=body)
            if r.ok:
                term_guid[tname] = r.json()["guid"]
                made += 1
    print(f"  glossary: {len(cat_guid)} categories, +{made} terms ({len(term_guid)} total)")
    return term_guid


def assign_terms(term_guid, base_tbl_guid):
    n = 0
    for tid, tguid in term_guid.items():
        eg = base_tbl_guid.get(tid)
        if not eg:
            continue
        r = api("POST", f"glossary/terms/{tguid}/assignedEntities",
                json=[{"guid": eg, "typeName": "hive_table"}])
        n += 1 if r.ok else 0
    print(f"  glossary: assigned {n} terms to tables")


# ── entities ─────────────────────────────────────────────────────────────────

def build_and_create(per_family):
    """Create dbs, base tables+columns, view tables+columns. Returns guid maps."""
    catalogs = {f: load_catalog(CATALOG_DIR / f"{f}.json") for f in FAMILIES}
    tables_by_fam = {}
    entities = []
    gid = itertools.count(1)

    def neg():
        return f"-{next(gid)}"

    # two dbs — create FIRST so every chunk can reference them by REAL guid
    gdb = _post_chunk([
        {"typeName": "hive_db", "guid": "-1", "attributes": {
            "qualifiedName": f"footprint@{CLUSTER}", "name": "footprint", "clusterName": CLUSTER,
            "description": "Ontology relational footprint — base tables (the larger warehouse the corpus views project from)"}},
        {"typeName": "hive_db", "guid": "-2", "attributes": {
            "qualifiedName": f"corpus@{CLUSTER}", "name": "corpus", "clusterName": CLUSTER,
            "description": "Textbook-corpus tables modeled as VIEWS over the footprint"}},
    ])
    db_fp, db_co = gdb.get("-1"), gdb.get("-2")
    print(f"  dbs: footprint={db_fp is not None}, corpus={db_co is not None}")

    base = {}   # template_id -> {"tref": neg, "cols": {slot: neg}, "fam":, "tmpl":}
    for f in FAMILIES:
        cat = catalogs[f]
        chosen = cat.templates[:per_family]
        tables_by_fam[f] = [{"template_id": t.template_id, "manchester": t.manchester_template} for t in chosen]
        for t in chosen:
            tref = neg()
            qn = f"footprint.{t.template_id}@{CLUSTER}"
            bfo = t.bfo_anchor_path[-1] if t.bfo_anchor_path else ""
            entities.append({"typeName": "hive_table", "guid": tref, "attributes": {
                "qualifiedName": qn, "name": t.template_id,
                "description": (t.verbal_template or t.manchester_template)[:500],
                "comment": t.manchester_template[:1000]},
                "relationshipAttributes": {"db": {"guid": db_fp}}})
            cols = {}
            for i, (slot, owl) in enumerate(t.slot_types.items()):
                cref = neg()
                entities.append({"typeName": "hive_column", "guid": cref, "attributes": {
                    "qualifiedName": f"footprint.{t.template_id}.{slot}@{CLUSTER}",
                    "name": slot, "type": owl, "position": i,
                    "comment": f"slot {slot}:{owl} of {t.template_id}"},
                    "relationshipAttributes": {"table": {"guid": tref}}})
                cols[slot] = cref
            base[t.template_id] = {"tref": tref, "cols": cols, "fam": f, "owl": dict(t.slot_types),
                                   "bfo": bfo, "complex": t.is_complex,
                                   "manchester": t.manchester_template}

    # synthesized views: per family, projections (subset of one base table) + a couple joins
    views = []  # (view_qn, view_tref, [source base trefs], [ (col_neg) ])
    for f in FAMILIES:
        fam_tids = [tid for tid, b in base.items() if b["fam"] == f]
        # 4 single-table projection views
        for tid in fam_tids[:4]:
            b = base[tid]
            vref = neg()
            vqn = f"corpus.view_{tid}@{CLUSTER}"
            entities.append({"typeName": "hive_table", "guid": vref, "attributes": {
                "qualifiedName": vqn, "name": f"view_{tid}",
                "description": f"Corpus view: projection over footprint.{tid}"},
                "relationshipAttributes": {"db": {"guid": db_co}}})
            proj = list(b["cols"])[: max(1, len(b["cols"]) - 1)]  # drop one col => "larger footprint"
            for slot in proj:
                entities.append({"typeName": "hive_column", "guid": neg(), "attributes": {
                    "qualifiedName": f"{vqn}.{slot}", "name": slot, "type": b["owl"][slot]},
                    "relationshipAttributes": {"table": {"guid": vref}}})
            views.append({"vref": vref, "src": [b["tref"]], "vqn": vqn})
        # 2 join views across two base tables of the family
        for a, c in list(zip(fam_tids, fam_tids[1:]))[:2]:
            ba, bc = base[a], base[c]
            vref = neg()
            vqn = f"corpus.join_{a}__{c}@{CLUSTER}"
            entities.append({"typeName": "hive_table", "guid": vref, "attributes": {
                "qualifiedName": vqn, "name": f"join_{a}__{c}",
                "description": f"Corpus view: join over footprint.{a} ⋈ footprint.{c} (family-simplex)"},
                "relationshipAttributes": {"db": {"guid": db_co}}})
            for src, lbl in ((ba, a), (bc, c)):
                for slot in list(src["cols"])[:2]:
                    entities.append({"typeName": "hive_column", "guid": neg(), "attributes": {
                        "qualifiedName": f"{vqn}.{lbl}_{slot}", "name": f"{lbl}_{slot}",
                        "type": src["owl"][slot]}, "relationshipAttributes": {"table": {"guid": vref}}})
            views.append({"vref": vref, "src": [ba["tref"], bc["tref"]], "vqn": vqn})

    print(f"  creating {len(entities)} entities in chunks (tables/columns/views) ...")
    ga = {}
    chunk = []
    for e in entities:
        if e["typeName"] == "hive_table" and len(chunk) >= 25:
            ga.update(_post_chunk(chunk)); chunk = []
        chunk.append(e)
    if chunk:
        ga.update(_post_chunk(chunk))
    print(f"   -> {len(ga)} guids assigned across chunks")

    base_tbl_guid = {tid: ga[b["tref"]] for tid, b in base.items() if b["tref"] in ga}
    # lineage processes (real guids)
    procs = []
    for v in views:
        out_g = ga.get(v["vref"])
        ins = [ga.get(s) for s in v["src"] if ga.get(s)]
        if not out_g or not ins:
            continue
        procs.append({"typeName": "Process",
            "attributes": {"qualifiedName": f"{v['vqn']}::derive", "name": "derive_view"},
            "relationshipAttributes": {
                "inputs": [{"guid": g, "typeName": "hive_table"} for g in ins],
                "outputs": [{"guid": out_g, "typeName": "hive_table"}]}})
    if procs:
        r = api("POST", "entity/bulk", json={"entities": procs}, timeout=600)
        print(f"  lineage: {len(procs)} processes -> {r.status_code}" + ("" if r.ok else f"  {r.text[:300]}"))

    return base, base_tbl_guid, {tid: {s: ga.get(g) for s, g in b["cols"].items()} for tid, b in base.items()}, tables_by_fam


# ── classifications + business metadata ──────────────────────────────────────

def classify(base, base_tbl_guid, base_col_guid):
    # group guids by classification type
    by_type: dict[str, list[str]] = {}
    bm_set = 0
    for tid, b in base.items():
        tg = base_tbl_guid.get(tid)
        if tg:
            by_type.setdefault(f"domain_{fam_label(b['fam'])}", []).append(tg)
            if b["bfo"] == "bfo:Process":
                by_type.setdefault("cpa_bfo_process", []).append(tg)
        for slot, owl in b["owl"].items():
            cg = base_col_guid.get(tid, {}).get(slot)
            if not cg:
                continue
            by_type.setdefault(f"cta_{owl.lower()}", []).append(cg)
            if b["complex"]:
                by_type.setdefault("cpa_complex_axiom", []).append(cg)

    applied = 0
    for tname, guids in by_type.items():
        # de-dup
        guids = list(dict.fromkeys(guids))
        r = api("POST", "entity/bulk/classification", json={
            "classification": {"typeName": tname, "attributes": {"confidence": "0.95",
                               "evidence": "ontology slot/anchor (paper-prototype retrofit)"}},
            "entityGuids": guids})
        if r.status_code in (200, 204):
            applied += len(guids)
        elif "ATLAS-409" in r.text or "already associated" in r.text:
            applied += len(guids)
        else:
            print(f"   classify {tname}: {r.status_code} {r.text[:160]}")
    print(f"  classifications: {applied} applications across {len(by_type)} types")

    # business metadata on base tables
    for tid, b in base.items():
        tg = base_tbl_guid.get(tid)
        if not tg:
            continue
        r = api("POST", f"entity/guid/{tg}/businessmetadata?isOverwrite=true", json={
            "OntologyProvenance": {"template_id": tid, "family": b["fam"], "bfo_anchor": b["bfo"],
                                   "manchester": b["manchester"][:1900], "is_complex": b["complex"]}})
        bm_set += 1 if r.ok else 0
    print(f"  business metadata: set on {bm_set} tables")


def reset():
    """Dump the prototype: delete corpus+footprint tables/columns/views/processes (by qn prefix)."""
    for tn, q in [("hive_column", "footprint."), ("hive_column", "corpus."),
                  ("hive_table", "footprint."), ("hive_table", "corpus."), ("Process", "corpus.")]:
        body = {"typeName": tn, "excludeDeletedEntities": True, "limit": 1000,
                "entityFilters": {"attributeName": "qualifiedName", "operator": "startsWith", "attributeValue": q}}
        res = api("POST", "search/basic", json=body)
        guids = [e["guid"] for e in res.json().get("entities", [])] if res.ok else []
        for i in range(0, len(guids), 50):
            api("DELETE", "entity/bulk", params=[("guid", g) for g in guids[i:i + 50]])
        print(f"  reset {tn} {q}*: {len(guids)} deleted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--reset", action="store_true", help="delete the projected prototype, then exit")
    args = ap.parse_args()

    if args.reset:
        print("RESET prototype:")
        reset()
        return

    print(f"Atlas relational projector — families={FAMILIES} per_family={args.per_family}")
    print("1) types"); ensure_types()
    print("2) entities + lineage")
    base, base_tbl_guid, base_col_guid, tables_by_fam = build_and_create(args.per_family)
    print("3) glossary")
    try:
        term_guid = ensure_glossary(tables_by_fam)
        assign_terms(term_guid, base_tbl_guid)
    except Exception as e:
        print(f"   glossary step failed (non-fatal): {e}")
    print("4) classifications + business metadata"); classify(base, base_tbl_guid, base_col_guid)
    print("done.")


if __name__ == "__main__":
    main()
