#!/usr/bin/env python
"""Materialize the corpus as topic-grounded COLLECTIONS into ``corpora/corpus/collections/``.

A **collection** = a FinePDFs-derived **topic** (carried forward — the cluster the chapters were
generated around) + its **chapters** (the prose + embedded views, as TEXT in the tree) + the
**relational tables underlying the views** those chapters embed (the polyglot DDL spine, with this
release's *semantic* columns) + the **ontology terms** grounding them + a **manifest** of the
cross-links. This is the collection-structured, chapters-in-tree release form of sdg-corpora —
the corpus content lives in the distribution; only the Atelier answer-key stays withheld
(a separate release artifact). Deterministic; no LLM / GPU / network.

    corpus/collections/topic-NNN-<family>/
      README.md          collection card (topic gist · chapters · terms · tables)
      chapters/<id>.md   chapter prose + embedded views (+ frontmatter)
      tables/<name>.sql  the underlying table's CREATE TABLE (semantic columns)
      manifest.json      cross-links (chapter→terms, topic→terms→tables)
    corpus/collections/INDEX.md   all collections + gap topics
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pyarrow.parquet as pq  # noqa: E402

from aegir.ontology import ddl as D  # noqa: E402
from aegir.ontology.complex import FamilyComplex  # noqa: E402
from aegir.ontology.schema import load_catalog  # noqa: E402

FAMILIES = ["01_foundation", "02_observation_measurement", "03_directive_governance",
            "04_ebpf_kernel", "05_provo_lineage", "06_belief_structure", "07_long_tail"]
CAT = REPO / "src/aegir/ontology/catalog"
FC = REPO / "src/aegir/ontology/family_complex.json"


def _slug(s: str, n: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s[:n] or "x"


def _cited(row: dict) -> list[str]:
    """The chapter's cited template_ids (stored as a stringified list)."""
    v = row.get("template_ids")
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            return [str(x) for x in ast.literal_eval(v)]
        except Exception:  # noqa: BLE001
            return []
    return []


def _styles(row: dict) -> list[int]:
    """The chapter's style topic ids — target + these = its full topic membership (the de-flatten)."""
    v = row.get("style_topic_ids")
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    if isinstance(v, str):
        try:
            return [int(x) for x in ast.literal_eval(v)]
        except Exception:  # noqa: BLE001
            return []
    return []


def _title(md: str) -> str | None:
    for ln in (md or "").splitlines():
        if ln.startswith("# "):
            return ln[2:].strip()
    return None


def _card(tid, tinfo, chs, terms, tables, tmpl, fam, topics=None, shared=None) -> str:
    gist = re.sub(r"\s+", " ", (tinfo.get("topic_repr_text") or "")).strip()[:400]
    topics = topics or [tid]
    shared = shared or {}
    L = [f"# Collection — topic {tid} · {fam or ''}", "",
         f"FinePDFs-grounded topic (carried forward from the coverage audit). "
         f"**{len(chs)} chapters** · **{len(topics)} topics** (target + style) · "
         f"**{len(terms)} ontology terms** · **{len(tables)} underlying tables**.", ""]
    if len(topics) > 1:
        L += ["**Topics (many-to-many).** The documents draw on, beyond the anchor topic "
              f"{tid}: " + ", ".join(f"topic {t}" for t in topics if t != tid) + ".", ""]
    if gist:
        L += ["> **Topic gist** (representative FinePDFs text): " + gist + " …", ""]
    L += ["## Chapters", ""]
    for r in chs:
        L.append(f"- [{_title(r.get('response_text') or '') or r['chapter_id']}]"
                 f"(chapters/{r['chapter_id']}.md)")
    L += ["", "## Ontology terms grounding this collection", ""]
    for tt in terms:
        _f, t = tmpl[tt]
        defn = re.sub(r"\s+", " ", (t.verbal_template or t.manchester_template or "")).strip()[:120]
        L.append(f"- `{tt}` — {defn}")
    L += ["", "## Underlying relational tables (semantic columns)", "",
          "_The tables the chapters' embedded views project from — the current semantic-column "
          "DDL spine. (Embedded views in the chapter prose reflect the generation-time schema.)_", ""]
    for st in tables:
        sh = shared.get(st.template.template_id, [])
        L.append(f"- [`{st.table.name}`](tables/{st.table.name}.sql) — realizes "
                 f"`{st.template.template_id}`"
                 + (f" · _shared with {len(sh)} other collection(s)_" if sh else ""))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="path to a corpus run's chapters.parquet")
    ap.add_argument("--coverage", required=True, help="path to the topic_coverage.parquet")
    ap.add_argument("--out", default=str(REPO / "corpora" / "corpus" / "collections"))
    args = ap.parse_args()

    chapters = pq.read_table(args.corpus).to_pylist()
    topics = {int(t["topic_id"]): t for t in pq.read_table(args.coverage).to_pylist()}

    tmpl: dict[str, tuple[str, object]] = {}
    spine: list = []
    for f in FAMILIES:
        for t in load_catalog(CAT / f"{f}.json").templates:
            tmpl[t.template_id] = (f, t)
            try:
                spine.append(D.template_to_table(t, f))
            except Exception:  # noqa: BLE001
                pass
    st_by_tid = {st.template.template_id: st for st in spine}
    try:
        fks, _ = D.cross_family_fks(spine, FamilyComplex.from_json(FC))
    except Exception:  # noqa: BLE001
        fks = []
    out_fks: dict[str, list] = defaultdict(list)
    for e in fks:
        out_fks[e.src_table].append(e)

    by_topic: dict[int, list] = defaultdict(list)
    for r in chapters:
        by_topic[int(r["target_topic_id"])].append(r)

    # Pre-pass for the de-flatten: per-collection terms + the term→collections inverse, so each
    # collection can show which of its base tables are SHARED with other collections (the
    # many-to-many footprint — a base table hydrates views across collections).
    coll_terms = {tid: sorted({tt for r in rs for tt in _cited(r) if tt in tmpl})
                  for tid, rs in by_topic.items()}
    term_colls: dict[str, list[int]] = defaultdict(list)
    for ctid, tms in coll_terms.items():
        for tt in tms:
            term_colls[tt].append(ctid)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index = []
    n_tables = 0
    for tid in sorted(by_topic):
        chs = sorted(by_topic[tid], key=lambda r: r["chapter_id"])
        tinfo = topics.get(tid, {})
        fam = tinfo.get("top_family") or chs[0].get("family")
        terms = coll_terms[tid]
        topics_full = sorted({tid} | {s for r in chs for s in _styles(r)})
        shared = {tt: sorted(set(term_colls[tt]) - {tid}) for tt in terms if len(term_colls[tt]) > 1}
        tables = [st_by_tid[tt] for tt in terms if tt in st_by_tid]
        cdir = out / f"topic-{tid:03d}-{_slug(fam.split('_', 1)[-1] if fam else 'x')}"
        (cdir / "chapters").mkdir(parents=True)
        (cdir / "tables").mkdir(parents=True)
        for r in chs:
            fm = (f"---\nchapter_id: {r['chapter_id']}\ntopic_id: {tid}\n"
                  f"family: {r.get('family')}\ncited_terms: {_cited(r)}\n"
                  f"model: {r.get('model')}\n---\n\n")
            (cdir / "chapters" / f"{r['chapter_id']}.md").write_text(fm + (r.get("response_text") or ""))
        for st in tables:
            (cdir / "tables" / f"{st.table.name}.sql").write_text(
                D.render_ddl(st, out_fks.get(st.table.name, [])) + "\n")
            n_tables += 1
        (cdir / "manifest.json").write_text(json.dumps({
            "topic_id": tid, "family": fam, "n_chapters": len(chs),
            "coverage_score": tinfo.get("coverage_score"),
            "topic_repr": (tinfo.get("topic_repr_text") or "")[:600],
            "terms": terms, "tables": [st.table.name for st in tables],
            "topics": topics_full, "shared_tables": shared,
            "chapters": [{"id": r["chapter_id"], "family": r.get("family"),
                          "topics": sorted({int(r["target_topic_id"])} | set(_styles(r))),
                          "cited_terms": _cited(r)} for r in chs],
        }, indent=2, default=str))
        (cdir / "README.md").write_text(_card(tid, tinfo, chs, terms, tables, tmpl, fam, topics_full, shared))
        index.append((tid, cdir.name, len(chs), len(terms), len(tables), fam))

    gaps = sorted(set(topics) - set(by_topic))
    L = ["# sdg-corpora — collections", "",
         f"{len(index)} populated collections ({len(chapters)} chapters · {n_tables} table DDLs) · "
         f"{len(gaps)} gap topics (no chapters yet). Each collection bundles a FinePDFs topic, its "
         "chapters (prose + embedded views), the underlying semantic-column tables, and the grounding "
         "ontology terms.", "", "## Collections", ""]
    for tid, dirn, nc, nt, ntab, fam in index:
        L.append(f"- [topic {tid}]({dirn}/README.md) — {fam or ''} · {nc} ch · {nt} terms · {ntab} tables")
    if gaps:
        L += ["", "## Gap topics (coverage holes — curation candidates, no chapters yet)", ""]
        for g in gaps:
            ti = topics.get(g, {})
            L.append(f"- topic {g} — {ti.get('top_family', '')} · _{ti.get('status', 'gap')}_")
    (out / "INDEX.md").write_text("\n".join(L) + "\n")

    print(f"collections: {len(index)} populated + {len(gaps)} gap topics → {out}")
    print(f"  {len(chapters)} chapters, {n_tables} table DDLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
