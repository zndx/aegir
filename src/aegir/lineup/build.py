"""``aegir.lineup build`` — project the three Data Products into the navigable KB.

Materializes the ontology / relational / content Data Products (data-mesh term of
art) into ``build/dev/current/`` as uniform KB notes (frontmatter + ``[[wikilinks]]``),
the read-only Lineup navigates. Pure/deterministic from the local catalog + DDL
lowering (ontology, relational); content/topics are projected from on-disk corpus /
coverage runs when present. ``current/`` is regenerable (cleared each build); ``scratch/``
and ``archive/`` (the maturity lifecycle) are preserved.

Navigation graph (the lineup edges):
    ontology/family  ──▶ ontology/template ──▶ ontology/anchor
    ontology/template ◀─▶ relational/table          (the ontology↔DDL pivot)
    content/chapter   ──▶ ontology/template, topic   (what a chapter realizes)
    topic             ──▶ ontology/template, ontology/family
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
        edges = [N.wl(f"ontology/family/{fam}"),
                 N.wl(f"relational/table/{_table_id(t.template_id)}", "DDL view")]
        if aid:
            edges.insert(1, N.wl(aid, anchor))
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
                "is_complex": bool(getattr(t, "is_complex", False)),
                "manchester_template": t.manchester_template,
                "provenance": dict(getattr(t, "provenance", {}) or {})}))

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
def _table_id(template_id: str) -> str:
    return f"rel_{template_id}"      # stable, slug-safe table id keyed to the template


def project_relational(rows: list[tuple[str, CatalogTemplate]]) -> list[N.Note]:
    from aegir.ontology.ddl import template_to_table
    out: list[N.Note] = []
    for fam, t in rows:
        try:
            st = template_to_table(t, fam)
        except Exception as e:                       # noqa: BLE001 — skip un-lowerable, keep going
            print(f"  [relational] skip {t.template_id}: {type(e).__name__}: {e}")
            continue
        cols = st.table.columns
        rid = f"relational/table/{_table_id(t.template_id)}"
        tid = f"ontology/template/{t.template_id}"
        rowsmd = "\n".join(f"| `{c.name}` | {c.slot_type} | {c.slot_ref} |" for c in cols)
        body = (
            f"**Realizes ontology template.** {N.wl(tid, t.template_id)}  "
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
    for r in recs:
        cid = str(r.get("chapter_id") or r.get("hx_exchange_id") or len(out))
        tids = list(r.get("template_ids") or [])
        topic = r.get("target_topic_id")
        cites = " · ".join(N.wl(f"ontology/template/{x}", x) for x in tids[:8]) or "—"
        head = f"**Cites templates.** {cites}"
        if topic is not None:
            head += f"  ·  **Topic.** {N.wl(f'topic/{int(topic)}')}"
        text = (r.get("response_text") or "")[:3000]
        out.append(N.Note(
            id=f"content/chapter/{cid}", title=f"chapter {cid}", kind="content-chapter",
            data_product="content", body=f"{head}\n\n{text}\n", frontmatter={
                "family": r.get("family"), "model": r.get("model"), "ablation": r.get("ablation"),
                "target_topic_id": topic, "template_ids": tids}))
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
    for r in recs:
        tid = int(r["topic_id"])
        fam, top = r.get("top_family"), r.get("top_template_id")
        edges = []
        if top:
            edges.append(f"**Nearest template.** {N.wl(f'ontology/template/{top}', top)}")
        if fam:
            edges.append(f"**Top family.** {N.wl(f'ontology/family/{fam}', fam)}")
        head = (f"**Status.** {r.get('status')} (coverage {r.get('coverage_score')})  ·  "
                + "  ·  ".join(edges))
        out.append(N.Note(
            id=f"topic/{tid}", title=f"topic {tid}", kind="topic", data_product="content",
            body=f"{head}\n\n{(r.get('topic_repr_text') or '')[:1500]}\n", frontmatter={
                "status": r.get("status"), "coverage_score": r.get("coverage_score"),
                "top_family": fam, "top_template_id": top}))
    return out


def run(args=None) -> int:
    kb = S.kb_dir()
    current = kb / "current"
    shutil.rmtree(current, ignore_errors=True)        # regenerable projection
    for d in (current, kb / "scratch", kb / "archive"):
        d.mkdir(parents=True, exist_ok=True)

    rows = S.load_ontology()
    notes = project_ontology(rows) + project_relational(rows)
    print(f"  ontology+relational: {len(notes)} notes from {len(rows)} templates")

    crun = S.corpus_run()
    if crun:
        c = project_content(crun)
        notes += c
        print(f"  content: {len(c)} chapter notes  ({crun})")
    else:
        print("  content: (no on-disk corpus run — skipped; set AEGIR_CORPUS_RUN to project)")

    cov = S.coverage_run()
    if cov:
        tp = project_topics(cov)
        notes += tp
        print(f"  topics:  {len(tp)} topic notes   ({cov})")
    else:
        print("  topics:  (no on-disk coverage run — skipped; set AEGIR_COVERAGE_RUN to project)")

    for n in notes:
        N.write_note(kb, n)
    idx = N.write_index(kb, notes)
    by_dp: dict[str, int] = {}
    for n in notes:
        by_dp[n.data_product] = by_dp.get(n.data_product, 0) + 1
    print(f"\n  KB projection → {kb}")
    print(f"  {len(notes)} notes  {by_dp}  ·  index {idx}")
    print("  roots: current (projection) | scratch (in-flux) | archive (superseded)")
    return 0
