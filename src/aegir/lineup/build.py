"""``aegir.lineup build`` — project the three Data Products into the navigable KB.

Materializes the ontology / relational / content Data Products (data-mesh term of
art) into ``build/dev/current/`` as uniform KB notes (frontmatter + ``[[wikilinks]]``)
the read-only Lineup navigates. Pure/deterministic from the local catalog + DDL
lowering (ontology, relational); content/topics are projected from on-disk corpus /
coverage runs when present. ``current/`` is regenerable (cleared each build); ``scratch/``
and ``archive/`` (the maturity lifecycle) are preserved.

Navigation graph (the lineup edges):
    lens/terms   ──▶ ontology/family  ──▶ ontology/template ──▶ ontology/anchor
    lens/schema  ──▶ relational/family ──▶ relational/table
    ontology/template ◀─▶ relational/table          (the ontology↔DDL pivot)
    lens/content ──▶ content/index ──▶ content/chapter ──▶ ontology/template, topic
                 └─▶ topic/index    ──▶ topic         ──▶ ontology/template, family

Lenses are the landing-page entry points (a way to understand a slice of the KB);
panels + links are the flexible substrate beneath them.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from aegir.lineup import notes as N
from aegir.lineup import sources as S
from aegir.ontology.schema import CatalogTemplate


def _verbal(t: CatalogTemplate) -> str:
    v = re.sub(r"\{[^}]+\}", "", t.verbal_template or "")
    v = re.sub(r"\s+", " ", v).strip(" .")
    return v or t.template_id.replace("_", " ")


def _anchor_id(anchor: str) -> str:
    return "ontology/anchor/" + re.sub(r"[^A-Za-z0-9_-]+", "_", anchor)


def _table_id(template_id: str) -> str:
    return f"rel_{template_id}"      # stable, slug-safe table id keyed to the template


# ── ontology Data Product ───────────────────────────────────────────────────
def project_ontology(rows: list[tuple[str, CatalogTemplate]]) -> list[N.Note]:
    fams: dict[str, list[str]] = {}
    anchors: dict[str, dict] = {}     # anchor_id -> {"label", "parent", "templates"}
    out: list[N.Note] = []

    for fam, t in rows:
        tid = f"ontology/template/{t.template_id}"
        fams.setdefault(fam, []).append(tid)
        path = list(t.bfo_anchor_path or [])
        anchor = path[-1] if path else None
        aid = _anchor_id(anchor) if anchor else None
        if aid and anchor:
            a = anchors.setdefault(aid, {"label": anchor, "parent": None, "templates": []})
            a["templates"].append(tid)
            if len(path) >= 2:
                a["parent"] = _anchor_id(path[-2])

        slots = ", ".join(f"`{s}` ({owl})" for s, owl in (t.slot_types or {}).items())
        body = (
            f"**Verbalization.** {_verbal(t)}\n\n"
            f"**Axiom (Manchester).** `{t.manchester_template}`\n\n"
            f"**Slots.** {slots or '—'}\n\n"
            f"**Family.** {N.wl(f'ontology/family/{fam}', fam)}"
            + (f"  ·  **BFO/CCO anchor.** {N.wl(aid, anchor)}" if aid else "")
            + f"  ·  **Relational projection.** {N.wl(f'relational/table/{_table_id(t.template_id)}', 'table')}\n"
        )
        out.append(N.Note(
            id=tid, title=t.template_id, kind="ontology-template", data_product="ontology",
            body=body, frontmatter={
                "family": fam, "bfo_anchor_path": path, "slot_types": dict(t.slot_types or {}),
                "is_complex": bool(t.is_complex), "manchester_template": t.manchester_template,
                "provenance": dict(t.provenance or {})}))

    for fam, tids in sorted(fams.items()):
        body = f"Catalog family **{fam}** — {len(tids)} templates.\n\n" + \
               "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(tids))
        out.append(N.Note(id=f"ontology/family/{fam}", title=fam, kind="ontology-family",
                          data_product="ontology", body=body, frontmatter={"n_templates": len(tids)}))

    for aid, a in sorted(anchors.items()):
        body = f"BFO/CCO anchor **{a['label']}** — {len(a['templates'])} templates anchored here.\n\n"
        if a["parent"]:
            body += f"**Broader.** {N.wl(a['parent'])}\n\n"
        body += "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(a["templates"]))
        out.append(N.Note(id=aid, title=a["label"], kind="ontology-anchor",
                          data_product="ontology", body=body,
                          frontmatter={"anchor": a["label"], "n_templates": len(a["templates"])}))
    return out


# ── relational Data Product ─────────────────────────────────────────────────
def project_relational(rows: list[tuple[str, CatalogTemplate]]) -> list[N.Note]:
    from aegir.ontology.ddl import template_to_table
    out: list[N.Note] = []
    fam_tables: dict[str, list[str]] = {}
    for fam, t in rows:
        try:
            st = template_to_table(t, fam)
        except Exception as e:                       # noqa: BLE001 — skip un-lowerable, keep going
            print(f"  [relational] skip {t.template_id}: {type(e).__name__}: {e}")
            continue
        cols = st.table.columns
        rid = f"relational/table/{_table_id(t.template_id)}"
        fam_tables.setdefault(fam, []).append(rid)
        tid = f"ontology/template/{t.template_id}"
        rowsmd = "\n".join(f"| `{c.name}` | {c.slot_type} | {c.slot_ref} |" for c in cols)
        body = (
            f"**Realizes ontology term.** {N.wl(tid, t.template_id)}  "
            f"(the ontology↔DDL pivot — DeepOnto semantics ↔ polyglot SQL syntax)\n\n"
            f"**Columns** ({len(cols)}).\n\n"
            f"| column | owl/sql type | slot |\n|---|---|---|\n{rowsmd}\n\n"
            f"_FK-following navigation (table → table along foreign keys) lands in U2 "
            f"(the relational DDL navigator)._\n"
        )
        out.append(N.Note(
            id=rid, title=st.table.name, kind="relational-table", data_product="relational",
            body=body, frontmatter={
                "family": fam, "realizes": t.template_id, "table_name": st.table.name,
                "columns": [{"name": c.name, "type": c.slot_type, "slot": c.slot_ref} for c in cols],
                "not_null": sorted(getattr(st, "not_null", set()) or set())}))

    for fam, rids in sorted(fam_tables.items()):
        body = f"Relational tables in family **{fam}** — {len(rids)}.\n\n" + \
               "\n".join(f"- {N.wl(rid, rid.split('/')[-1])}" for rid in sorted(rids))
        out.append(N.Note(id=f"relational/family/{fam}", title=f"{fam} (tables)",
                          kind="relational-family", data_product="relational", body=body,
                          frontmatter={"n_tables": len(rids)}))
    return out


# ── content Data Product (on-disk corpus run, if present) ────────────────────
def project_content(run: Path) -> list[N.Note]:
    import pyarrow.parquet as pq
    out: list[N.Note] = []
    try:
        recs = pq.read_table(run).to_pylist()
    except Exception as e:                            # noqa: BLE001
        print(f"  [content] could not read {run}: {type(e).__name__}: {e}")
        return out
    chapter_ids: list[str] = []
    for r in recs:
        cid = str(r.get("chapter_id") or r.get("hx_exchange_id") or len(out))
        chapter_ids.append(cid)
        tids = list(r.get("template_ids") or [])
        topic = r.get("target_topic_id")
        cites = " · ".join(N.wl(f"ontology/template/{x}", x) for x in tids[:8]) or "—"
        head = f"**Cites terms.** {cites}"
        if topic is not None:
            head += f"  ·  **Topic.** {N.wl(f'topic/{int(topic)}')}"
        text = (r.get("response_text") or "")[:3000]
        out.append(N.Note(
            id=f"content/chapter/{cid}", title=f"chapter {cid}", kind="content-chapter",
            data_product="content", body=f"{head}\n\n{text}\n", frontmatter={
                "family": r.get("family"), "model": r.get("model"), "ablation": r.get("ablation"),
                "target_topic_id": topic, "template_ids": tids}))
    body = (f"**Corpus** — {len(chapter_ids)} chapters.\n\n"
            + "\n".join(f"- {N.wl(f'content/chapter/{c}', f'chapter {c}')}" for c in chapter_ids)
            + f"\n\nFinePDFs topics: {N.wl('topic/index')}\n")
    out.append(N.Note(id="content/index", title="Corpus", kind="content-index",
                      data_product="content", body=body, frontmatter={"n_chapters": len(chapter_ids)}))
    return out


# ── topics (coverage audit, if present) — bridge content↔ontology ────────────
def project_topics(run: Path) -> list[N.Note]:
    import pyarrow.parquet as pq
    out: list[N.Note] = []
    try:
        recs = pq.read_table(run).to_pylist()
    except Exception as e:                            # noqa: BLE001
        print(f"  [topics] could not read {run}: {type(e).__name__}: {e}")
        return out
    by_status: dict[str, list[int]] = {}
    for r in recs:
        tid = int(r["topic_id"])
        by_status.setdefault(str(r.get("status")), []).append(tid)
        fam, top = r.get("top_family"), r.get("top_template_id")
        edges = []
        if top:
            edges.append(f"**Nearest term.** {N.wl(f'ontology/template/{top}', top)}")
        if fam:
            edges.append(f"**Top family.** {N.wl(f'ontology/family/{fam}', fam)}")
        head = (f"**Status.** {r.get('status')} (coverage {r.get('coverage_score')})  ·  "
                + "  ·  ".join(edges))
        out.append(N.Note(
            id=f"topic/{tid}", title=f"topic {tid}", kind="topic", data_product="content",
            body=f"{head}\n\n{(r.get('topic_repr_text') or '')[:1500]}\n", frontmatter={
                "status": r.get("status"), "coverage_score": r.get("coverage_score"),
                "top_family": fam, "top_template_id": top}))
    parts = [f"**Topics** — {sum(len(v) for v in by_status.values())} FinePDFs topics."]
    for status, ids in sorted(by_status.items()):
        parts.append(f"\n**{status}** ({len(ids)}).\n" +
                     "\n".join(f"- {N.wl(f'topic/{i}', f'topic {i}')}" for i in sorted(ids)))
    out.append(N.Note(id="topic/index", title="Topics", kind="topic-index",
                      data_product="content", body="\n".join(parts),
                      frontmatter={"by_status": {k: len(v) for k, v in by_status.items()}}))
    return out


# ── lenses (the landing-page entry points) ───────────────────────────────────
def project_lenses(families: list[str], has_content: bool, has_topics: bool) -> list[N.Note]:
    def lens(key, dp, title, intro, links):
        body = f"**{title}.** {intro}\n\n" + "\n".join(f"- {x}" for x in links)
        return N.Note(id=f"lens/{key}", title=title, kind="lens", data_product=dp,
                      body=body, frontmatter={"lens": key})
    terms = lens("terms", "ontology", "Terms",
                 "The ontology vocabulary, by family. Open a family to browse its terms; each "
                 "term shows its verbalization, axiom, BFO/CCO anchor, and its relational projection.",
                 [N.wl(f"ontology/family/{f}", f) for f in families])
    schema = lens("schema", "relational", "Schema",
                  "The relational projection, by family. Each table realizes one ontology term "
                  "(the ontology↔DDL pivot); columns carry typed slots.",
                  [N.wl(f"relational/family/{f}", f) for f in families])
    clinks = []
    if has_content:
        clinks.append(N.wl("content/index", "the corpus chapters"))
    if has_topics:
        clinks.append(N.wl("topic/index", "the FinePDFs topics"))
    content = lens("content", "content", "Content",
                   "The textbook-quality corpus and the FinePDFs topics it covers." if clinks
                   else "No corpus/topics projected yet (run a corpus/coverage build first).",
                   clinks)
    return [terms, schema, content]


def run(args=None) -> int:
    kb = S.kb_dir()
    current = kb / "current"
    shutil.rmtree(current, ignore_errors=True)        # regenerable projection
    for d in (current, kb / "scratch", kb / "archive"):
        d.mkdir(parents=True, exist_ok=True)

    rows = S.load_ontology()
    families = sorted({fam for fam, _ in rows})
    notes = project_ontology(rows) + project_relational(rows)
    print(f"  ontology+relational: {len(notes)} notes from {len(rows)} templates")

    has_content = has_topics = False
    crun = S.corpus_run()
    if crun:
        c = project_content(crun)
        notes += c
        has_content = bool(c)
        print(f"  content: {len(c)} notes  ({crun})")
    else:
        print("  content: (no on-disk corpus run — skipped; set AEGIR_CORPUS_RUN to project)")

    cov = S.coverage_run()
    if cov:
        tp = project_topics(cov)
        notes += tp
        has_topics = bool(tp)
        print(f"  topics:  {len(tp)} notes   ({cov})")
    else:
        print("  topics:  (no on-disk coverage run — skipped; set AEGIR_COVERAGE_RUN to project)")

    notes += project_lenses(families, has_content, has_topics)

    for n in notes:
        N.write_note(kb, n)
    idx = N.write_index(kb, notes)
    by_dp: dict[str, int] = {}
    for n in notes:
        by_dp[n.data_product] = by_dp.get(n.data_product, 0) + 1
    print(f"\n  KB projection → {kb}")
    print(f"  {len(notes)} notes  {by_dp}  ·  index {idx}")
    print("  lenses: lens/terms · lens/schema · lens/content")
    print("  roots: current (projection) | scratch (in-flux) | archive (superseded)")
    return 0
