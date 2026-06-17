#!/usr/bin/env python3
"""Atlas DDL-native projector (supersedes project_atlas_relational.py).

Projects the ontology's relational footprint into Apache Atlas v2 as the honest
**rdbms_*** model (no Hive), driven by the SQL DDL spine (``aegir.ontology.ddl``):

  - footprint = ``rdbms_db`` of ``rdbms_table`` (type=TABLE) from the DDL spine;
    columns are ``rdbms_column`` with data_type / isPrimaryKey / isNullable read
    from the deterministic lowering (and the DDL text is validated by polyglot —
    the projection is gated on real SQL validity).
  - cross-family joins = ``rdbms_foreign_key`` entities (the family-complex-
    sanctioned join structure the corpus views exploit).
  - corpus = ``rdbms_table`` (type=VIEW) over the footprint; per-view SQL is run
    through ``polyglot_sql.openlineage_run_event`` and ingested via
    ``aegir.governance.ol`` → **column-level** lineage (view.col ← base.col) in aegir_hx.
  - classifications (CTA per column OWL type, domain per family, CPA complex),
    business metadata (OntologyProvenance), glossary (per-family categories,
    per-template terms) — as before, retyped to rdbms_*.

Degrades gracefully if polyglot isn't built yet (skips validation + column lineage;
entities/classifications/BM/glossary still project). Idempotent (upsert by qn).

Run:  uv run --no-sync python scripts/project_atlas_ddl.py [--per-family 8] [--reset]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.governance import ol  # noqa: E402
from aegir.ontology import ddl as D  # noqa: E402
from aegir.ontology.complex import FamilyComplex  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402
from aegir.lineup.build import _axiom_kind, _term_vocab, _verbal  # noqa: E402  (SKOS helpers)

ATLAS = "http://127.0.0.1:21000"
CLUSTER = "aegir"
CATALOG_DIR = REPO / "src/aegir/ontology/catalog"
FC_PATH = REPO / "src/aegir/ontology/family_complex.json"
FAMILIES = ["01_foundation", "02_observation_measurement", "03_directive_governance",
            "04_ebpf_kernel", "05_provo_lineage", "06_belief_structure", "07_long_tail"]

S = requests.Session()
S.auth = ("admin", "admin")
S.headers["Content-Type"] = "application/json"


def api(method: str, path: str, timeout: int = 120, **kw):
    return S.request(method, f"{ATLAS}/api/atlas/v2/{path}", timeout=timeout, **kw)


def _post_chunk(ents):
    r = api("POST", "entity/bulk", json={"entities": ents}, timeout=600)
    if not r.ok:
        print("   chunk FAILED:", r.status_code, r.text[:300])
        return {}
    return r.json().get("guidAssignments", {})


def fam_label(f: str) -> str:
    return f.split("_", 1)[1]


# ── type system ──────────────────────────────────────────────────────────────
OWL_TYPES = ["Class", "ObjectProperty", "DataProperty", "Individual", "Datatype", "AnnotationProperty"]


def _attr(name, tn="string", **opts):
    a = {"name": name, "typeName": tn, "isOptional": True, "cardinality": "SINGLE",
         "valuesMinCount": 0, "valuesMaxCount": 1, "isUnique": False, "isIndexable": True}
    a.update(opts)
    return a


def ensure_types():
    existing = {h["name"] for h in api("GET", "types/typedefs/headers").json()}
    # A prior hive run may have created OntologyProvenance with applicableEntityTypes=hive_*,
    # which 404s on rdbms_table. Force-refresh it so it targets rdbms_table/rdbms_column.
    if "OntologyProvenance" in existing:
        d = api("GET", "types/businessmetadatadef/name/OntologyProvenance")
        if not (d.ok and "rdbms_table" in d.text):  # stale hive-only def → refresh for rdbms_*
            api("DELETE", "types/typedef/name/OntologyProvenance")
            existing.discard("OntologyProvenance")
    class_defs, bm_defs = [], []

    def cdef(name, desc):
        return {"name": name, "description": desc, "superTypes": [],
                "attributeDefs": [_attr("confidence"), _attr("evidence")]}

    for o in OWL_TYPES:
        if (n := f"cta_{o.lower()}") not in existing:
            class_defs.append(cdef(n, f"CTA: column whose ontology slot type is OWL {o}"))
    for f in FAMILIES:
        if (n := f"domain_{fam_label(f)}") not in existing:
            class_defs.append(cdef(n, f"Domain: table realized from ontology family {f}"))
    for n, d in [("cpa_complex_axiom", "CPA: column from a complex (DeepOnto) axiom template"),
                 ("cpa_bfo_process", "CPA: table anchored under bfo:Process")]:
        if n not in existing:
            class_defs.append(cdef(n, d))

    if "OntologyProvenance" not in existing:
        def bm_attr(name, tn="string"):
            return _attr(name, tn, options={
                "applicableEntityTypes": "[\"rdbms_table\",\"rdbms_column\"]",
                "maxStrLength": "2000"})
        bm_defs.append({"name": "OntologyProvenance",
                        "description": "Provenance from the grounding ontology",
                        "attributeDefs": [bm_attr("template_id"), bm_attr("family"),
                                          bm_attr("bfo_anchor"), bm_attr("manchester"),
                                          bm_attr("is_complex", "boolean")]})

    if class_defs or bm_defs:
        r = api("POST", "types/typedefs", json={
            "classificationDefs": class_defs, "businessMetadataDefs": bm_defs,
            "enumDefs": [], "structDefs": [], "entityDefs": [], "relationshipDefs": []})
        print(f"  types: +{len(class_defs)} classifications, +{len(bm_defs)} businessMetadata -> {r.status_code}")
        if not r.ok:
            print("   ", r.text[:400])
    else:
        print("  types: already present")


# ── spine (DDL) ──────────────────────────────────────────────────────────────
def build_spine(per_family):
    spine = []
    for f in FAMILIES:
        cat = load_catalog(CATALOG_DIR / f"{f}.json")
        for t in cat.templates[:per_family]:
            spine.append(D.template_to_table(t, f))
    fks, _ = D.cross_family_fks(spine, FamilyComplex.from_json(FC_PATH))
    n_valid = 0
    try:
        for st in spine:
            st_fks = [e for e in fks if e.src_table == st.table.name]
            if all(d.valid for d in D.validate_ddl(D.render_ddl(st, st_fks))):
                n_valid += 1
        print(f"  spine: {len(spine)} tables, {len(fks)} cross-family FKs, "
              f"validated {n_valid}/{len(spine)} (trino∩spark)")
    except RuntimeError as e:
        print(f"  spine: {len(spine)} tables, {len(fks)} cross-family FKs "
              f"(validation skipped — {str(e)[:60]})")
    return spine, fks


# ── entities ─────────────────────────────────────────────────────────────────
def build_and_create(per_family):
    spine, fks = build_spine(per_family)
    by_tname = {st.table.name: st for st in spine}
    gid = itertools.count(1)
    neg = lambda: f"-{next(gid)}"  # noqa: E731

    inst = _post_chunk([{"typeName": "rdbms_instance", "guid": "-1", "attributes": {
        "qualifiedName": f"aegir@{CLUSTER}", "name": "aegir", "rdbms_type": "iceberg",
        "platform": "aegir / polyglot-verified DDL"}}]).get("-1")
    gdb = _post_chunk([
        {"typeName": "rdbms_db", "guid": "-1", "attributes": {
            "qualifiedName": f"footprint@{CLUSTER}", "name": "footprint",
            "comment": "Ontology relational footprint — base tables (the warehouse the corpus views project from)"},
            "relationshipAttributes": {"instance": {"guid": inst}}},
        {"typeName": "rdbms_db", "guid": "-2", "attributes": {
            "qualifiedName": f"corpus@{CLUSTER}", "name": "corpus",
            "comment": "Textbook-corpus tables modeled as VIEWS over the footprint"},
            "relationshipAttributes": {"instance": {"guid": inst}}},
    ])
    db_fp, db_co = gdb.get("-1"), gdb.get("-2")
    print(f"  instance={inst is not None} dbs: footprint={db_fp is not None} corpus={db_co is not None}")

    entities = []
    base = {}  # template_id -> {tref, cols{colname:cref}, st}
    for st in spine:
        tref = neg()
        qn = f"footprint.{st.table.name}@{CLUSTER}"
        entities.append({"typeName": "rdbms_table", "guid": tref, "attributes": {
            "qualifiedName": qn, "name": st.table.name, "type": "TABLE", "contact_info": "aegir",
            "comment": D.build_table_comment(st.template, st.family)[:1000]},
            "relationshipAttributes": {"db": {"guid": db_fp}}})
        cols = {}
        for cd in D.column_dicts(st):
            cref = neg()
            entities.append({"typeName": "rdbms_column", "guid": cref, "attributes": {
                "qualifiedName": f"footprint.{st.table.name}.{cd['name']}@{CLUSTER}",
                "name": cd["name"], "data_type": cd["sql_type"],
                "isPrimaryKey": cd["pk"], "isNullable": cd["nullable"]},
                "relationshipAttributes": {"table": {"guid": tref}}})
            cols[cd["name"]] = cref
        base[st.template.template_id] = {"tref": tref, "cols": cols, "st": st}

    # synthesized corpus views (projection + family-simplex join), with view SQL for lineage
    views = []
    by_fam = {}
    for st in spine:
        by_fam.setdefault(st.family, []).append(st)
    for sts in by_fam.values():
        for st in sts[:3]:
            b = base[st.template.template_id]
            vref = neg()
            vqn = f"corpus.view_{st.table.name}@{CLUSTER}"
            proj = list(b["cols"])[:max(1, len(b["cols"]) - 1)]
            entities.append({"typeName": "rdbms_table", "guid": vref, "attributes": {
                "qualifiedName": vqn, "name": f"view_{st.table.name}", "type": "VIEW", "contact_info": "aegir",
                "comment": f"corpus view: projection over footprint.{st.table.name}"},
                "relationshipAttributes": {"db": {"guid": db_co}}})
            for c in proj:
                entities.append({"typeName": "rdbms_column", "guid": neg(), "attributes": {
                    "qualifiedName": f"{vqn}.{c}", "name": c, "data_type": "VARCHAR(255)"},
                    "relationshipAttributes": {"table": {"guid": vref}}})
            sql = f"SELECT {', '.join(proj)} FROM footprint.{st.table.name}"
            views.append({"vref": vref, "name": f"view_{st.table.name}", "sql": sql})
        for sa, sc in list(zip(sts, sts[1:]))[:2]:
            a, c = sa.table.name, sc.table.name
            vref = neg()
            vqn = f"corpus.join_{a}__{c}@{CLUSTER}"
            ca = list(base[sa.template.template_id]["cols"])[:2]
            cc = list(base[sc.template.template_id]["cols"])[:2]
            entities.append({"typeName": "rdbms_table", "guid": vref, "attributes": {
                "qualifiedName": vqn, "name": f"join_{a}__{c}", "type": "VIEW", "contact_info": "aegir",
                "comment": f"corpus view: join footprint.{a} ⋈ footprint.{c} (family-simplex)"},
                "relationshipAttributes": {"db": {"guid": db_co}}})
            sel = [f"a.{x} AS a_{x}" for x in ca] + [f"c.{x} AS c_{x}" for x in cc]
            for x in ca:
                entities.append({"typeName": "rdbms_column", "guid": neg(), "attributes": {
                    "qualifiedName": f"{vqn}.a_{x}", "name": f"a_{x}", "data_type": "VARCHAR(255)"},
                    "relationshipAttributes": {"table": {"guid": vref}}})
            for x in cc:
                entities.append({"typeName": "rdbms_column", "guid": neg(), "attributes": {
                    "qualifiedName": f"{vqn}.c_{x}", "name": f"c_{x}", "data_type": "VARCHAR(255)"},
                    "relationshipAttributes": {"table": {"guid": vref}}})
            sql = (f"SELECT {', '.join(sel)} "
                   f"FROM footprint.{a} a JOIN footprint.{c} c ON a.id = c.id")
            views.append({"vref": vref, "name": f"join_{a}__{c}", "sql": sql})

    print(f"  creating {len(entities)} entities (rdbms tables/columns/views) in chunks ...")
    ga = {}
    chunk = []
    for e in entities:
        if e["typeName"] == "rdbms_table" and len(chunk) >= 25:
            ga.update(_post_chunk(chunk))
            chunk = []
        chunk.append(e)
    if chunk:
        ga.update(_post_chunk(chunk))
    print(f"   -> {len(ga)} guids assigned across chunks")

    # ── rdbms_foreign_key entities (2nd pass: real guids) ──
    fk_ents = []
    for e in fks:
        sb = by_tname.get(e.src_table)
        db_ = by_tname.get(e.dst_table)
        if not sb or not db_:
            continue
        bs, bd = base[sb.template.template_id], base[db_.template.template_id]
        src_tg, dst_tg = ga.get(bs["tref"]), ga.get(bd["tref"])
        src_cg, dst_cg = ga.get(bs["cols"].get(e.src_col)), ga.get(bd["cols"].get(e.dst_col))
        if not all([src_tg, dst_tg, src_cg, dst_cg]):
            continue
        fk_ents.append({"typeName": "rdbms_foreign_key", "guid": neg(), "attributes": {
            "qualifiedName": f"footprint.{e.src_table}.{e.src_col}__fk__{e.dst_table}@{CLUSTER}",
            "name": f"{e.src_table}.{e.src_col} -> {e.dst_table}.id"},
            "relationshipAttributes": {
                "table": {"guid": src_tg}, "key_columns": [{"guid": src_cg}],
                "references_table": {"guid": dst_tg}, "references_columns": [{"guid": dst_cg}]}})
    if fk_ents:
        # Chunk + non-fatal: at full-footprint scale a single bulk FK POST exceeds the 600s read
        # timeout and (when it raised) aborted the whole projection before glossary/assign/classify.
        ok = 0
        for i in range(0, len(fk_ents), 100):
            batch = fk_ents[i:i + 100]
            try:
                r = api("POST", "entity/bulk", json={"entities": batch}, timeout=300)
                ok += len(batch) if r.ok else 0
                if not r.ok:
                    print(f"   FK chunk [{i}:{i + len(batch)}] -> {r.status_code} {r.text[:120]}")
            except Exception as ex:  # noqa: BLE001 — FK creation must never abort the projection
                print(f"   FK chunk [{i}:{i + len(batch)}] failed (non-fatal): {type(ex).__name__}")
        print(f"  foreign keys: {ok}/{len(fk_ents)} rdbms_foreign_key created (chunked)")

    # ── column-level lineage via polyglot OpenLineage → ol.ingest_run_event ──
    n_lin = 0
    try:
        import datetime as _dt
        import uuid as _uuid

        import polyglot_sql as pg
        evt_time = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for v in views:
            try:
                evt = pg.openlineage_run_event(v["sql"], {
                    "producer": ol.PRODUCER, "datasetNamespace": "footprint",
                    "outputDataset": {"namespace": "corpus", "name": v["name"]},
                    "jobNamespace": "aegir", "jobName": "ddl_project",
                    "eventType": "COMPLETE", "eventTime": evt_time,
                    "runId": str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"ddl-project:{v['name']}"))})
                payload = evt["event"] if isinstance(evt, dict) and "event" in evt else evt
                n_lin += ol.ingest_run_event(payload).get("columns", 0)
            except Exception as ex:
                print(f"   lineage[{v['name']}] skipped: {str(ex)[:90]}")
        print(f"  column lineage: {n_lin} DERIVES_FROM edges from {len(views)} views")
    except ImportError:
        print("  column lineage: skipped (polyglot_sql not built)")

    return base, {tid: ga.get(b["tref"]) for tid, b in base.items()}, \
        {tid: {c: ga.get(g) for c, g in b["cols"].items()} for tid, b in base.items()}


# ── glossary ─────────────────────────────────────────────────────────────────
GLOSSARY_TIMEOUT = 25  # Atlas-on-AGE glossary term-traversal can hang; fail fast, don't block.


def glossary_responsive(probe_timeout: int = GLOSSARY_TIMEOUT) -> bool:
    """Probe the glossary term-traversal path before a full sync. On the AGE backend these
    queries (``/detailed``, ``/terms``) can hang 30-45s+; degrade gracefully (skip — the lineup
    carries the hierarchy) rather than hang the projector."""
    try:
        gr = api("GET", "glossary", timeout=15)
        glos = gr.json() if gr.ok and gr.text.strip() else []
        g = next((x for x in glos if x.get("name") == "Aegir Ontology"), None)
        if not g:
            return True  # no glossary yet — the create path is fine
        return api("GET", f"glossary/{g['guid']}/terms?limit=1", timeout=probe_timeout).ok
    except Exception:
        return False


# BFO/CCO upper-type leaf anchors → readable glossary sub-category names. The bfo_anchor_path
# leaf is one of these 7; it gives each family a second organizing axis (domain × upper-type).
_ANCHOR_LABELS = {
    "cco:Artifact": "Artifacts",
    "bfo:Process": "Processes",
    "cco:DescriptiveICE": "Descriptive Information",
    "cco:DirectiveICE": "Directive Information",
    "cco:DesignativeICE": "Designative Information",
    "cco:InformationContentEntity": "Information Content Entities",
    "bfo:IndependentContinuant": "Independent Continuants",
}


def _anchor_label(anchor):
    if not anchor:
        return None
    return _ANCHOR_LABELS.get(anchor, anchor.split(":")[-1])


def _term_anchor_label(t):
    path = t.bfo_anchor_path or []
    return _anchor_label(path[-1]) if path else None


def _load_categories(gguid):
    """Build idempotent category maps from /glossary/{g}/categories (full objects with name +
    parentCategory — the one glossary endpoint that renders these reliably on AGE). Returns
    (top: name→guid for un-parented family categories, sub: (parent_name, name)→guid for nested)."""
    r = api("GET", f"glossary/{gguid}/categories", timeout=90)
    cats = r.json() if r.ok and r.text.strip() else []
    if not isinstance(cats, list):
        cats = []
    by_guid = {c.get("guid"): c for c in cats}
    top, sub = {}, {}
    for c in cats:
        pg = (c.get("parentCategory") or {}).get("categoryGuid")
        if pg and pg in by_guid:
            sub[(by_guid[pg].get("name"), c.get("name"))] = c.get("guid")
        elif not pg:
            top[c.get("name")] = c.get("guid")
    return top, sub


def sync_category_hierarchy(gguid, all_terms):
    """Two-level category tree: 7 family categories (top) → BFO-anchor sub-categories (leaf),
    nested via parentCategory. Idempotent. Returns (top, sub, term_leaf: tid→leaf-category guid
    the term should be filed under — its family's anchor sub-cat, or the family itself if the
    template carries no BFO anchor)."""
    top, sub = _load_categories(gguid)
    for f in FAMILIES:                                   # 1) family top categories
        cname = fam_label(f)
        if cname not in top:
            r = api("POST", "glossary/category", json={"name": cname, "anchor": {"glossaryGuid": gguid}})
            if r.ok:
                top[cname] = r.json().get("guid")
    needed, fam_anchor = set(), {}                       # 2) which (family, anchor) sub-cats exist in the data
    for fam, t in all_terms:
        cname, alabel = fam_label(fam), _term_anchor_label(t)
        fam_anchor[t.template_id] = (cname, alabel)
        if alabel:
            needed.add((cname, alabel))
    for cname, alabel in sorted(needed):                 # 3) create missing sub-cats under their family
        if (cname, alabel) not in sub and cname in top:
            r = api("POST", "glossary/category", json={
                "name": alabel, "anchor": {"glossaryGuid": gguid},
                "parentCategory": {"categoryGuid": top[cname]}})
            if r.ok:
                sub[(cname, alabel)] = r.json().get("guid")
    term_leaf = {}                                       # 4) resolve each term's leaf category
    for tid, (cname, alabel) in fam_anchor.items():
        term_leaf[tid] = (sub.get((cname, alabel)) if alabel else None) or top.get(cname)
    n_sub = len({k for k in sub})
    print(f"  glossary: {len(top)} family categories, {n_sub} BFO sub-categories (2-level hierarchy)")
    return top, sub, term_leaf


def ensure_glossary(all_terms):
    gr = api("GET", "glossary", timeout=15)
    glos = gr.json() if gr.ok and gr.text.strip() else []
    g = next((x for x in glos if x.get("name") == "Aegir Ontology"), None)
    if g is None:
        g = api("POST", "glossary", json={"name": "Aegir Ontology",
                "shortDescription": "BFO/CCO-grounded ontology terms backing the relational footprint"}).json()
    gguid = g["guid"]
    # Use the base glossary object (term HEADERS: guid + displayText) rather than /detailed (full
    # term bodies). The headers are all we need to build the guid map, and on the AGE backend
    # /detailed materializes every term body (~28s, exceeds GLOSSARY_TIMEOUT) while this is ~half.
    dr = api("GET", f"glossary/{gguid}", timeout=90)
    detail = dr.json() if dr.ok and dr.text.strip() else {}
    term_guid = {t["displayText"]: t["termGuid"] for t in detail.get("terms", [])}

    top, sub, term_leaf = sync_category_hierarchy(gguid, all_terms)
    made = 0
    for _fam, t in all_terms:
        tid = t.template_id
        if tid in term_guid:
            continue
        leaf = term_leaf.get(tid)
        r = api("POST", "glossary/term", json={
            "name": tid, "anchor": {"glossaryGuid": gguid},
            "shortDescription": t.manchester_template[:240],
            "categories": [{"categoryGuid": leaf}] if leaf else []})
        if r.ok:
            term_guid[tid] = r.json()["guid"]
            made += 1
    print(f"  glossary: {len(top)} families + {len(sub)} sub-categories, +{made} terms "
          f"({len(term_guid)} total)")
    return term_guid, term_leaf


def enrich_and_link(term_guid, all_terms, term_leaf=None):
    """Sync the ontology SoT INTO the glossary terms: SKOS annotations as term attributes
    (definition → short/longDescription, the scope note → usage, the BERTSubs surface-form
    set → longDescription) + the verified ``broader`` subsumption hierarchy as Atlas ``isA``
    relationships (child ``isA`` its parents; Atlas auto-maintains the inverse ``classifies``
    on each parent, so the hierarchy is navigable BOTH ways in the glossary). When ``term_leaf``
    is given (tid→leaf-category guid), also re-files each term into its BFO sub-category so the
    2-level hierarchy applies to terms that already existed. GET-merge-PUT so assignedEntities
    survive. Idempotent — re-syncs to the current ontology."""
    term_leaf = term_leaf or {}
    n_skos = n_isa = n_fail = misses = 0
    for fam, t in all_terms:
        tguid = term_guid.get(t.template_id)
        if not tguid:
            continue
        try:
            cur = api("GET", f"glossary/term/{tguid}", timeout=15)
            if not cur.ok:
                n_fail += 1
                continue
            term = cur.json()
            pref = t.template_id.replace("_", " ")
            definition = _verbal(t) or t.manchester_template
            alts = [a for a in _term_vocab(t) if a != pref][:16]
            anchor = t.bfo_anchor_path[-1] if t.bfo_anchor_path else "the upper ontology"
            term["shortDescription"] = definition[:250]
            term["longDescription"] = (
                f"{definition}\n\nAxiom (Manchester): {t.manchester_template}"
                + (f"\n\nSurface forms (SKOS altLabel / retrieval): {', '.join(alts)}" if alts else ""))
            term["usage"] = (f"{_axiom_kind(t.manchester_template).capitalize()} anchored to "
                             f"{anchor}, in the '{fam_label(fam)}' category.")
            leaf = term_leaf.get(t.template_id)
            if leaf:
                term["categories"] = [{"categoryGuid": leaf}]   # re-file into the BFO sub-category
            has_isa = False
            parents = [term_guid[p] for p in (t.broader or []) if p in term_guid]
            if parents:
                term["isA"] = [{"termGuid": pg} for pg in parents]
                has_isa = True
            # CHECK the PUT — a non-2xx (slow AGE write path / validation) must NOT count as a
            # success. The prior bug incremented n_skos before an unchecked PUT, so it counted
            # attempts (540) while the writes silently failed (0 persisted) during the pre-index
            # slow window. Count only what actually persists.
            put = api("PUT", f"glossary/term/{tguid}", json=term, timeout=30)
            if put.ok:
                n_skos += 1
                if has_isa:
                    n_isa += 1
                misses = 0
            else:
                n_fail += 1
                misses += 1
                if n_fail <= 5:
                    print(f"   enrich {t.template_id}: PUT {put.status_code} {put.text[:120]}")
                if misses >= 5:
                    print(f"   enrich: aborting after {n_skos} persisted — Atlas glossary term "
                          "PUT repeatedly failing. The lineup carries the hierarchy.")
                    break
        except Exception as e:  # noqa: BLE001 — one bad term shouldn't abort the sync
            misses += 1
            n_fail += 1
            if misses >= 5:
                print(f"   enrich: aborting after {n_skos} persisted — Atlas glossary term API "
                      "repeatedly timing out (AGE perf). The lineup carries the hierarchy.")
                break
            print(f"   enrich {t.template_id}: {type(e).__name__}: {str(e)[:80]}")
    tail = f", {n_fail} failed" if n_fail else ""
    print(f"  glossary: enriched {n_skos} terms (SKOS), linked {n_isa} via isA (broader hierarchy){tail}")


def assign_terms(term_guid, base_tbl_guid):
    n = 0
    for tid, tguid in term_guid.items():
        eg = base_tbl_guid.get(tid)
        if eg:
            r = api("POST", f"glossary/terms/{tguid}/assignedEntities",
                    json=[{"guid": eg, "typeName": "rdbms_table"}])
            n += 1 if r.ok else 0
    print(f"  glossary: assigned {n} terms to tables")


# ── classifications + business metadata ──────────────────────────────────────
def classify(base, base_tbl_guid, base_col_guid):
    by_type: dict[str, list[str]] = {}
    for tid, b in base.items():
        st = b["st"]
        bfo = st.template.bfo_anchor_path[-1] if st.template.bfo_anchor_path else ""
        if (tg := base_tbl_guid.get(tid)):
            by_type.setdefault(f"domain_{fam_label(st.family)}", []).append(tg)
            if bfo == "bfo:Process":
                by_type.setdefault("cpa_bfo_process", []).append(tg)
        slot2col = {c.slot_ref: c.name for c in st.table.columns}  # slot_ref is canonical (cols renamed)
        for slot, owl in st.template.slot_types.items():
            if owl == "ObjectProperty":
                continue
            cg = base_col_guid.get(tid, {}).get(slot2col.get(slot, ""))
            if not cg:
                continue
            by_type.setdefault(f"cta_{owl.lower()}", []).append(cg)
            if st.template.is_complex:
                by_type.setdefault("cpa_complex_axiom", []).append(cg)

    applied = 0
    for tname, guids in by_type.items():
        guids = list(dict.fromkeys(guids))
        for i in range(0, len(guids), 50):
            batch = guids[i:i + 50]
            r = api("POST", "entity/bulk/classification", timeout=600, json={
                "classification": {"typeName": tname, "attributes": {
                    "confidence": "0.95", "evidence": "ontology slot/anchor (DDL spine)"}},
                "entityGuids": batch})
            if r.status_code in (200, 204) or "ATLAS-409" in r.text or "already associated" in r.text:
                applied += len(batch)
            else:
                print(f"   classify {tname}: {r.status_code} {r.text[:160]}")
    print(f"  classifications: {applied} applications across {len(by_type)} types")

    bm_set = 0
    for tid, b in base.items():
        tg = base_tbl_guid.get(tid)
        if not tg:
            continue
        st = b["st"]
        bfo = st.template.bfo_anchor_path[-1] if st.template.bfo_anchor_path else ""
        r = api("POST", f"entity/guid/{tg}/businessmetadata?isOverwrite=true", timeout=600, json={
            "OntologyProvenance": {"template_id": tid, "family": st.family, "bfo_anchor": bfo,
                                   "manchester": st.template.manchester_template[:1900],
                                   "is_complex": st.template.is_complex}})
        bm_set += 1 if r.ok else 0
    print(f"  business metadata: set on {bm_set} tables")


def reset():
    for tn, q in [("rdbms_column", "footprint."), ("rdbms_column", "corpus."),
                  ("rdbms_foreign_key", "footprint."),
                  ("rdbms_table", "footprint."), ("rdbms_table", "corpus."),
                  ("hive_column", "footprint."), ("hive_column", "corpus."),
                  ("hive_table", "footprint."), ("hive_table", "corpus."), ("Process", "corpus.")]:
        body = {"typeName": tn, "excludeDeletedEntities": True, "limit": 1000,
                "entityFilters": {"attributeName": "qualifiedName", "operator": "startsWith",
                                  "attributeValue": q}}
        res = api("POST", "search/basic", json=body)
        guids = [e["guid"] for e in res.json().get("entities", [])] if res.ok else []
        for i in range(0, len(guids), 50):
            api("DELETE", "entity/bulk", params=[("guid", g) for g in guids[i:i + 50]])
        if guids:
            print(f"  reset {tn} {q}*: {len(guids)} deleted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--glossary-only", action="store_true",
                    help="re-sync only the glossary (all terms + SKOS + broader→isA); "
                         "skip rdbms/classifications. Idempotent.")
    ap.add_argument("--assign-only", action="store_true",
                    help="link every term to its already-projected table (assignedEntities), "
                         "discovering table guids by qualifiedName. Recovery for an interrupted "
                         "full projection; no entity creation. Idempotent.")
    args = ap.parse_args()

    if args.reset:
        print("RESET projection:")
        reset()
        return

    if args.assign_only:
        print("ASSIGN-ONLY: link all terms to their projected footprint tables:")
        if not glossary_responsive():
            print("   glossary unresponsive — aborting."); return
        all_terms = [(f, t) for f in FAMILIES
                     for t in load_catalog(CATALOG_DIR / f"{f}.json").templates]
        tables, off = {}, 0          # discover footprint tables by qualifiedName (one search, not 540 GETs)
        while True:
            r = api("POST", "search/basic", json={
                "typeName": "rdbms_table", "excludeDeletedEntities": True,
                "attributes": ["qualifiedName"], "limit": 1000, "offset": off}, timeout=120)
            ents = r.json().get("entities", []) if r.ok else []
            for e in ents:
                qn = (e.get("attributes") or {}).get("qualifiedName")
                if qn:
                    tables[qn] = e["guid"]
            if len(ents) < 1000:
                break
            off += 1000
        base_tbl_guid = {}
        for f, t in all_terms:
            qn = f"footprint.{D.template_to_table(t, f).table.name}@{CLUSTER}"
            if qn in tables:
                base_tbl_guid[t.template_id] = tables[qn]
        print(f"   discovered {len(tables)} footprint tables; matched "
              f"{len(base_tbl_guid)}/{len(all_terms)} terms→tables")
        tg, _ = ensure_glossary(all_terms)
        assign_terms(tg, base_tbl_guid)
        print("done.")
        return

    if args.glossary_only:
        print("GLOSSARY-ONLY re-sync (terms + SKOS + broader→isA):")
        if not glossary_responsive():
            print("   glossary unresponsive — aborting."); return
        all_terms = [(f, t) for f in FAMILIES
                     for t in load_catalog(CATALOG_DIR / f"{f}.json").templates]
        tg, term_leaf = ensure_glossary(all_terms)
        enrich_and_link(tg, all_terms, term_leaf)
        print("done.")
        return

    print(f"Atlas DDL-native projector — families={len(FAMILIES)} per_family={args.per_family}")
    print("1) types"); ensure_types()
    print("2) entities (rdbms) + foreign keys + column lineage")
    base, base_tbl_guid, base_col_guid = build_and_create(args.per_family)
    print("3) glossary (all terms + SKOS + broader→isA hierarchy)")
    if not glossary_responsive():
        print("   glossary SKIPPED — Atlas-on-AGE glossary term-traversal is unresponsive "
              "(/detailed, /terms time out). The lineup (build/dev) carries the broader/narrower "
              "hierarchy + SKOS; re-run once Atlas glossary perf is addressed.")
    else:
        try:
            all_terms = [(f, t) for f in FAMILIES
                         for t in load_catalog(CATALOG_DIR / f"{f}.json").templates]
            tg, term_leaf = ensure_glossary(all_terms)  # 540 terms + BFO category hierarchy (per-family-independent)
            enrich_and_link(tg, all_terms, term_leaf)    # SKOS + broader→isA + re-file into sub-categories
            assign_terms(tg, base_tbl_guid)              # link projected tables to their terms (assignedEntities)
        except Exception as e:
            print(f"   glossary step failed (non-fatal): {e}")
    print("4) classifications + business metadata"); classify(base, base_tbl_guid, base_col_guid)
    print("done.")


if __name__ == "__main__":
    main()
