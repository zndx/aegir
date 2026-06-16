"""``aegir.lineup build`` — project the three Data Products into the navigable KB.

Materializes the ontology / relational / content Data Products (data-mesh term of
art) into ``build/dev/current/`` as uniform KB notes (frontmatter + ``[[wikilinks]]``)
the read-only Lineup navigates. Pure/deterministic from the local catalog + DDL
lowering (ontology, relational); content/topics are projected from on-disk corpus /
coverage runs when present. ``current/`` is regenerable (cleared each build); ``scratch/``
and ``archive/`` (the maturity lifecycle) are preserved.

Organization is Atlas-glossary aligned (Lexicon / Category / Term). Edges are dense
enough to CURATE: beyond the spine (Lexicon→Category→Term→anchor / ontology↔DDL pivot),
the build computes CROSS-REFERENCES so a Term shows what exercises it —
    Term ──realized by──▶ content/chapter        (which chapters instantiate it)
    Term ──mapped from──▶ topic                   (which FinePDFs topics it's nearest to)
    topic ──covered by──▶ content/chapter         (coverage; "(gap)" if none)
— turning the thin star-graph into a web you can navigate for curation.
"""
from __future__ import annotations

import re
import shutil

from aegir.lineup import notes as N
from aegir.lineup import sources as S
from aegir.ontology.schema import CatalogTemplate

LEXICON = "aegir"      # the single Lexicon (Atlas Glossary) for now; extend when multi-domain


def _verbal(t: CatalogTemplate) -> str:
    v = re.sub(r"\{[^}]+\}", "", t.verbal_template or "")
    v = re.sub(r"\s+", " ", v).strip(" .")
    return v or t.template_id.replace("_", " ")


def _anchor_id(anchor: str) -> str:
    return "ontology/anchor/" + re.sub(r"[^A-Za-z0-9_-]+", "_", anchor)


def _table_id(template_id: str) -> str:
    return f"rel_{template_id}"


def _chapter_id(r: dict, i: int) -> str:
    return str(r.get("chapter_id") or r.get("hx_exchange_id") or f"idx{i}")


def category_qn(name: str, parent_qn: str | None = None) -> str:
    return f"{name}.{parent_qn}" if parent_qn else f"{name}@{LEXICON}"


# ── ontology Data Product — the Lexicon (Terms organized by Categories) ──────
def project_ontology(rows: list[tuple[str, CatalogTemplate]],
                     term_chapters: dict[str, list[str]] | None = None,
                     term_topics: dict[str, list[int]] | None = None) -> list[N.Note]:
    term_chapters = term_chapters or {}
    term_topics = term_topics or {}
    cats: dict[str, list[str]] = {}
    anchors: dict[str, dict] = {}
    out: list[N.Note] = []

    for cat, t in rows:
        term_id = f"ontology/term/{t.template_id}"
        cat_id = f"ontology/category/{cat}"
        cats.setdefault(cat, []).append(term_id)
        path = list(t.bfo_anchor_path or [])
        anchor = path[-1] if path else None
        aid = _anchor_id(anchor) if anchor else None
        if aid and anchor:
            a = anchors.setdefault(aid, {"label": anchor, "parent": None, "terms": []})
            a["terms"].append(term_id)
            if len(path) >= 2:
                a["parent"] = _anchor_id(path[-2])

        slots = ", ".join(f"`{s}` ({owl})" for s, owl in (t.slot_types or {}).items())
        body = (
            f"**Verbalization.** {_verbal(t)}\n\n"
            f"**Axiom (Manchester).** `{t.manchester_template}`\n\n"
            f"**Slots.** {slots or '—'}\n\n"
            f"**Category.** {N.wl(cat_id, cat)}"
            + (f"  ·  **BFO/CCO anchor.** {N.wl(aid, anchor)}" if aid else "")
            + f"  ·  **Relational projection.** {N.wl(f'relational/table/{_table_id(t.template_id)}', 'table')}\n"
        )
        # Cross-references — what exercises this term (the curation surface).
        chs = term_chapters.get(t.template_id, [])
        tps = sorted(term_topics.get(t.template_id, []))
        if chs:
            body += ("\n**Realized by** " + str(len(chs)) + " chapter(s): "
                     + ", ".join(N.wl(f"content/chapter/{c}", c) for c in chs[:8])
                     + (" …" if len(chs) > 8 else "") + "\n")
        if tps:
            body += ("\n**Mapped from** " + str(len(tps)) + " topic(s): "
                     + ", ".join(N.wl(f"topic/{tp}", f"topic {tp}") for tp in tps[:12])
                     + (" …" if len(tps) > 12 else "") + "\n")
        if not chs and not tps:
            body += "\n_Not yet exercised by any chapter or topic (a curation candidate)._\n"

        out.append(N.Note(
            id=term_id, title=t.template_id, kind="ontology-term", data_product="ontology",
            body=body, frontmatter={
                "qualified_name": f"{t.template_id}@{LEXICON}", "category": cat,
                "category_qualified_name": category_qn(cat),
                "bfo_anchor_path": path, "slot_types": dict(t.slot_types or {}),
                "is_complex": bool(t.is_complex), "manchester_template": t.manchester_template,
                "n_chapters": len(chs), "n_topics": len(tps),
                "provenance": dict(t.provenance or {})}))

    for cat, term_ids in sorted(cats.items()):
        qn = category_qn(cat)
        body = (
            f"**Category** `{qn}` — {len(term_ids)} terms. Part of the {N.wl('lens/terms', 'Lexicon')}.\n\n"
            "A category organizes terms so the term's context can be enriched.\n\n"
            + "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(term_ids))
        )
        out.append(N.Note(id=f"ontology/category/{cat}", title=cat, kind="ontology-category",
                          data_product="ontology", body=body, frontmatter={
                              "qualified_name": qn, "parent": None, "children": [], "n_terms": len(term_ids)}))

    for aid, a in sorted(anchors.items()):
        body = f"BFO/CCO anchor **{a['label']}** — {len(a['terms'])} terms anchored here (siblings).\n\n"
        if a["parent"]:
            body += f"**Broader.** {N.wl(a['parent'])}\n\n"
        body += "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(a["terms"]))
        out.append(N.Note(id=aid, title=a["label"], kind="ontology-anchor",
                          data_product="ontology", body=body,
                          frontmatter={"anchor": a["label"], "n_terms": len(a["terms"])}))
    return out


# ── relational Data Product ─────────────────────────────────────────────────
def project_relational(rows: list[tuple[str, CatalogTemplate]]) -> list[N.Note]:
    from aegir.ontology.ddl import template_to_table
    out: list[N.Note] = []
    cat_tables: dict[str, list[str]] = {}
    for cat, t in rows:
        try:
            st = template_to_table(t, cat)
        except Exception as e:                       # noqa: BLE001
            print(f"  [relational] skip {t.template_id}: {type(e).__name__}: {e}")
            continue
        cols = st.table.columns
        rid = f"relational/table/{_table_id(t.template_id)}"
        cat_tables.setdefault(cat, []).append(rid)
        term_id = f"ontology/term/{t.template_id}"
        rowsmd = "\n".join(f"| `{c.name}` | {c.slot_type} | {c.slot_ref} |" for c in cols)
        body = (
            f"**Realizes term.** {N.wl(term_id, t.template_id)}  "
            f"(the ontology↔DDL pivot — DeepOnto semantics ↔ polyglot SQL syntax)\n\n"
            f"**Columns** ({len(cols)}).\n\n"
            f"| column | owl/sql type | slot |\n|---|---|---|\n{rowsmd}\n\n"
            f"_FK-following navigation (table → table along foreign keys) lands in U2 "
            f"(the relational DDL navigator)._\n"
        )
        out.append(N.Note(
            id=rid, title=st.table.name, kind="relational-table", data_product="relational",
            body=body, frontmatter={
                "category": cat, "realizes": t.template_id, "table_name": st.table.name,
                "columns": [{"name": c.name, "type": c.slot_type, "slot": c.slot_ref} for c in cols],
                "not_null": sorted(getattr(st, "not_null", set()) or set())}))

    for cat, rids in sorted(cat_tables.items()):
        body = f"Relational tables in category **{cat}** — {len(rids)}.\n\n" + \
               "\n".join(f"- {N.wl(rid, rid.split('/')[-1])}" for rid in sorted(rids))
        out.append(N.Note(id=f"relational/category/{cat}", title=f"{cat} (tables)",
                          kind="relational-category", data_product="relational", body=body,
                          frontmatter={"category": cat, "n_tables": len(rids)}))
    return out


# ── content Data Product (on-disk corpus rows, if present) ───────────────────
def project_content(recs: list[dict]) -> list[N.Note]:
    out: list[N.Note] = []
    chapter_ids: list[str] = []
    for i, r in enumerate(recs):
        cid = _chapter_id(r, i)
        chapter_ids.append(cid)
        tids = list(r.get("template_ids") or [])
        topic = r.get("target_topic_id")
        cites = " · ".join(N.wl(f"ontology/term/{x}", x) for x in tids[:8]) or "—"
        head = f"**Realizes terms.** {cites}"
        if topic is not None:
            head += f"  ·  **Topic.** {N.wl(f'topic/{int(topic)}')}"
        text = (r.get("response_text") or "")[:3000]
        out.append(N.Note(
            id=f"content/chapter/{cid}", title=f"chapter {cid}", kind="content-chapter",
            data_product="content", body=f"{head}\n\n{text}\n", frontmatter={
                "category": r.get("family"), "model": r.get("model"), "ablation": r.get("ablation"),
                "target_topic_id": topic, "template_ids": tids}))
    body = (f"**Corpus** — {len(chapter_ids)} chapters.\n\n"
            + "\n".join(f"- {N.wl(f'content/chapter/{c}', f'chapter {c}')}" for c in chapter_ids)
            + f"\n\nFinePDFs topics: {N.wl('topic/index')}\n")
    out.append(N.Note(id="content/index", title="Corpus", kind="content-index",
                      data_product="content", body=body, frontmatter={"n_chapters": len(chapter_ids)}))
    return out


# ── topics (coverage rows, if present) — bridge content↔ontology ─────────────
def project_topics(recs: list[dict], topic_chapters: dict[int, list[str]] | None = None) -> list[N.Note]:
    topic_chapters = topic_chapters or {}
    out: list[N.Note] = []
    by_status: dict[str, list[int]] = {}
    for r in recs:
        tid = int(r["topic_id"])
        by_status.setdefault(str(r.get("status")), []).append(tid)
        cat, top = r.get("top_family"), r.get("top_template_id")
        edges = []
        if top:
            edges.append(f"**Nearest term.** {N.wl(f'ontology/term/{top}', top)}")
        if cat:
            edges.append(f"**Top category.** {N.wl(f'ontology/category/{cat}', cat)}")
        covered = topic_chapters.get(tid, [])
        cov = (f"**Covered by** {len(covered)} chapter(s): "
               + ", ".join(N.wl(f"content/chapter/{c}", c) for c in covered[:6])) if covered else "**Covered by** 0 chapters (gap)"
        head = (f"**Status.** {r.get('status')} (coverage {r.get('coverage_score')})  ·  "
                + "  ·  ".join([*edges, cov]))
        out.append(N.Note(
            id=f"topic/{tid}", title=f"topic {tid}", kind="topic", data_product="content",
            body=f"{head}\n\n{(r.get('topic_repr_text') or '')[:1500]}\n", frontmatter={
                "status": r.get("status"), "coverage_score": r.get("coverage_score"),
                "top_category": cat, "top_term": top, "n_chapters": len(covered)}))
    parts = [f"**Topics** — {sum(len(v) for v in by_status.values())} FinePDFs topics."]
    for status, ids in sorted(by_status.items()):
        parts.append(f"\n**{status}** ({len(ids)}).\n" +
                     "\n".join(f"- {N.wl(f'topic/{i}', f'topic {i}')}" for i in sorted(ids)))
    out.append(N.Note(id="topic/index", title="Topics", kind="topic-index",
                      data_product="content", body="\n".join(parts),
                      frontmatter={"by_status": {k: len(v) for k, v in by_status.items()}}))
    return out


# ── lenses (the landing-page entry points) ───────────────────────────────────
def project_lenses(categories: list[str], has_content: bool, has_topics: bool) -> list[N.Note]:
    terms = N.Note(
        id="lens/terms", title="Lexicon", kind="lens", data_product="ontology",
        frontmatter={"lens": "terms", "lexicon": LEXICON},
        body=("**Lexicon** `" + LEXICON + "`. A lexicon includes terms — a term is a useful word "
              "for the enterprise. A category organizes terms so the term's context can be enriched. "
              "Browse by category:\n\n"
              + "\n".join(f"- {N.wl(f'ontology/category/{c}', c)}" for c in categories)))
    schema = N.Note(
        id="lens/schema", title="Schema", kind="lens", data_product="relational",
        frontmatter={"lens": "schema", "lexicon": LEXICON},
        body=("**Schema.** The relational projection of the lexicon — each table realizes one term "
              "(the ontology↔DDL pivot); columns carry typed slots. By category:\n\n"
              + "\n".join(f"- {N.wl(f'relational/category/{c}', c)}" for c in categories)))
    clinks = []
    if has_content:
        clinks.append(N.wl("content/index", "the corpus chapters"))
    if has_topics:
        clinks.append(N.wl("topic/index", "the FinePDFs topics"))
    content = N.Note(
        id="lens/content", title="Content", kind="lens", data_product="content",
        frontmatter={"lens": "content"},
        body=("**Content.** The textbook-quality corpus and the FinePDFs topics it covers.\n\n"
              + ("\n".join(f"- {x}" for x in clinks) if clinks
                 else "No corpus/topics projected yet (run a corpus/coverage build first).")))
    return [terms, schema, content]


def run(args=None) -> int:
    kb = S.kb_dir()
    current = kb / "current"
    shutil.rmtree(current, ignore_errors=True)        # regenerable projection
    for d in (current, kb / "scratch", kb / "archive"):
        d.mkdir(parents=True, exist_ok=True)

    rows = S.load_ontology()
    categories = sorted({cat for cat, _ in rows})

    # Read content/coverage once + derive cross-references (so Terms show what exercises them).
    corpus, crun = S.corpus_recs()
    coverage, cov = S.coverage_recs()
    term_chapters: dict[str, list[str]] = {}
    topic_chapters: dict[int, list[str]] = {}
    for i, r in enumerate(corpus):
        cid = _chapter_id(r, i)
        for tid in (r.get("template_ids") or []):
            term_chapters.setdefault(tid, []).append(cid)
        tp = r.get("target_topic_id")
        if tp is not None:
            topic_chapters.setdefault(int(tp), []).append(cid)
    term_topics: dict[str, list[int]] = {}
    for r in coverage:
        top = r.get("top_template_id")
        if top:
            term_topics.setdefault(top, []).append(int(r["topic_id"]))

    notes = (project_ontology(rows, term_chapters, term_topics) + project_relational(rows))
    print(f"  ontology+relational: {len(notes)} notes from {len(rows)} terms "
          f"in {len(categories)} categories (Lexicon {LEXICON!r})")

    if corpus:
        c = project_content(corpus)
        notes += c
        print(f"  content: {len(c)} notes  ({crun})")
    else:
        print("  content: (no on-disk corpus run — skipped; set AEGIR_CORPUS_RUN to project)")
    if coverage:
        tp = project_topics(coverage, topic_chapters)
        notes += tp
        print(f"  topics:  {len(tp)} notes   ({cov})")
    else:
        print("  topics:  (no on-disk coverage run — skipped; set AEGIR_COVERAGE_RUN to project)")

    notes += project_lenses(categories, bool(corpus), bool(coverage))

    for n in notes:
        N.write_note(kb, n)
    entries = (N.scan_notes(kb, "current") + N.scan_notes(kb, "scratch") + N.scan_notes(kb, "archive"))
    idx = N.write_index(kb, entries)
    by_dp: dict[str, int] = {}
    by_root: dict[str, int] = {}
    edges = 0
    for e in entries:
        by_dp[e["data_product"]] = by_dp.get(e["data_product"], 0) + 1
        by_root[e["root"]] = by_root.get(e["root"], 0) + 1
        edges += len(e["links"])
    print(f"\n  KB projection → {kb}")
    print(f"  {len(entries)} notes  {edges} edges  by_dp={by_dp}  by_root={by_root}  ·  index {idx}")
    print("  lenses: lens/terms (Lexicon) · lens/schema · lens/content")
    print("  roots: current (projection, alongside) | scratch (authored) | archive (aged + snapshots)")
    return 0
