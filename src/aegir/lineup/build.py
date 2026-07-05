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

import json
import re
import shutil
from pathlib import Path

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


def _chapter_title(cid: str, reg: str | None = None) -> str:
    """Human label for a chapter link. Refined chapters encode their primary template + register in the id
    (``ch_live_<template>.<register>``) → humanize it; c3 chapters carry a content hash → short-id fallback."""
    body = cid
    for rg in (".natural", ".semantic"):
        if body.endswith(rg):
            body, reg = body[: -len(rg)], reg or rg[1:]
            break
    if body.startswith("ch_live_"):              # strip the trailing _<hash6> uniqueness suffix from the title
        name = re.sub(r"_[0-9a-f]{6}$", "", body[len("ch_live_"):]).replace("_", " ").strip().title() or f"ch {cid[:8]}"
    else:
        name = f"ch {cid[:8]}"
    return f"{name} · {reg}" if reg else name


_BOILER = frozenset(
    "the a an of for to and or is are be class subclassof some only exactly min max value "
    "equivalentto disjointwith disjointclasses subclass entity thing relation property type kind "
    "via with has that this these those such each its their which when where also into not".split())
_WORD = re.compile(r"[a-z][a-z0-9]+")


def _tokens(s: str) -> list[str]:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", s).replace("_", " ").lower()
    return [w for w in _WORD.findall(spaced) if w not in _BOILER and len(w) > 2]


def _term_vocab(t: CatalogTemplate) -> list[str]:
    """The term's distinguishing vocabulary — id tokens + slot names + verbalization content,
    minus boilerplate. Seeds skos:altLabel (the BERTSubs multi-label set / MaxSim surfaces);
    curation refines it toward real synonym phrases. (Converge with coverage_r1.construct_terms.)"""
    toks = set(_tokens(t.template_id))
    for slot in (t.slot_types or {}):
        toks.update(_tokens(slot))
    toks.update(_tokens(re.sub(r"\{[^}]+\}", " ", t.verbal_template or "")))
    return sorted(toks)


def _axiom_kind(m: str) -> str:
    if "EquivalentTo" in m:
        return "equivalence"
    if " some " in m:
        return "existential restriction"
    if " only " in m:
        return "universal restriction"
    if any(k in m for k in (" min ", " max ", " exactly ")):
        return "cardinality restriction"
    if "Disjoint" in m:
        return "disjointness axiom"
    if "SubClassOf" in m:
        return "subsumption"
    return "axiom"


def category_qn(name: str, parent_qn: str | None = None) -> str:
    return f"{name}.{parent_qn}" if parent_qn else f"{name}@{LEXICON}"


# ── ontology Data Product — the Lexicon (Terms organized by Categories) ──────
def project_ontology(rows: list[tuple[str, CatalogTemplate]],
                     term_chapters: dict[str, list[str]] | None = None,
                     term_topics: dict[str, list[int]] | None = None,
                     term_broader: dict[str, list[str]] | None = None,
                     term_narrower: dict[str, list[str]] | None = None,
                     term_colls: dict[str, list[str]] | None = None) -> list[N.Note]:
    term_chapters = term_chapters or {}
    term_topics = term_topics or {}
    term_broader = term_broader or {}
    term_narrower = term_narrower or {}
    term_colls = term_colls or {}
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
        cls = term_colls.get(t.template_id, [])
        if cls:
            body += ("\n**Realized across** " + str(len(cls)) + " collection(s): "
                     + ", ".join(N.wl(c, c.split("-")[-1]) for c in cls[:12])
                     + (" …" if len(cls) > 12 else "") + "\n")
        if not chs and not tps:
            body += "\n_Not yet exercised by any chapter or topic (a curation candidate)._\n"

        # Subsumption hierarchy — the autonomously-mediated, HermiT-verified is-a edges (U1).
        # Per-term navigation: walk Broader (this ⊑ parent) ↑ and Narrower (child ⊑ this) ↓.
        bro = term_broader.get(t.template_id, [])
        nar = sorted(term_narrower.get(t.template_id, []))
        if bro or nar:
            body += "\n**Hierarchy** (HermiT-verified subsumption):\n"
            if bro:
                body += ("\n- **Broader** — this ⊑ "
                         + ", ".join(N.wl(f"ontology/term/{x}", x) for x in bro) + "\n")
            if nar:
                body += ("\n- **Narrower** — ⊑ this: "
                         + ", ".join(N.wl(f"ontology/term/{x}", x) for x in nar[:12])
                         + (f" … (+{len(nar) - 12})" if len(nar) > 12 else "") + "\n")

        # Retrieval annotations (SKOS) — common constructs with domain values, recorded as
        # ontology annotation properties (the principled home; the panel, ColBERT/MaxSim, and
        # BERTSubs all read them from here). skos:altLabel is the BERTSubs multi-label set.
        pref = t.template_id.replace("_", " ")
        skos = {
            "skos:prefLabel": pref,
            "skos:altLabel": [a for a in _term_vocab(t) if a != pref][:24],
            "skos:definition": _verbal(t),
            "skos:scopeNote": (f"{_axiom_kind(t.manchester_template).capitalize()} anchored to "
                               f"{anchor or 'the upper ontology'}, in the '{cat}' category."),
            "skos:example": [f"chapter {c}" for c in chs[:3]],
        }
        rtext = " ".join([skos["skos:prefLabel"], *skos["skos:altLabel"], skos["skos:definition"],
                          skos["skos:scopeNote"], *skos["skos:example"]])
        body += (
            "\n**Retrieval annotations** (SKOS — the ColBERT/MaxSim + BERTSubs label-set):\n\n"
            f"- **skos:prefLabel** — {skos['skos:prefLabel']}\n"
            f"- **skos:altLabel** — {', '.join(skos['skos:altLabel']) or '—'}\n"
            f"- **skos:definition** — {skos['skos:definition']}\n"
            f"- **skos:scopeNote** — {skos['skos:scopeNote']}\n"
            + (f"- **skos:example** — {', '.join(skos['skos:example'])}\n" if skos["skos:example"] else "")
        )

        out.append(N.Note(
            id=term_id, title=t.template_id, kind="ontology-term", data_product="ontology",
            body=body, frontmatter={
                "qualified_name": f"{t.template_id}@{LEXICON}", "category": cat,
                "category_qualified_name": category_qn(cat),
                "bfo_anchor_path": path, "slot_types": dict(t.slot_types or {}),
                "is_complex": bool(t.is_complex), "manchester_template": t.manchester_template,
                "n_chapters": len(chs), "n_topics": len(tps),
                "broader": bro, "narrower": nar,
                "skos": skos, "retrieval_text": rtext,
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
    """Project the DDL spine as relational/table notes (U2: FK-following navigation).

    Each table carries its typed columns, its outgoing **Foreign keys** (a column →
    the referenced table, navigable), and the inverse **Referenced by**, so the lineup
    panel-trail walks the relational graph along foreign keys — the invention."""
    import math
    from collections import Counter

    from aegir.ontology.ddl import cross_family_fks, template_to_table

    # Build the full spine first so cross-family FKs (the family-complex-gated joins) resolve.
    spine = []
    for cat, t in rows:
        try:
            spine.append(template_to_table(t, cat))
        except Exception as e:                       # noqa: BLE001
            print(f"  [relational] skip {t.template_id}: {type(e).__name__}: {e}")
    try:
        import aegir.ontology as _onto
        from aegir.ontology.complex import FamilyComplex
        fc = FamilyComplex.from_json(Path(_onto.__file__).resolve().parent / "family_complex.json")
        fks, _audit = cross_family_fks(spine, fc)
    except Exception as e:                            # noqa: BLE001 — notes are valid without FKs
        print(f"  [relational] FK graph skipped: {type(e).__name__}: {e}")
        fks = []
    name2tid = {st.table.name: st.template.template_id for st in spine}
    out_fks: dict[str, list] = {}
    in_fks: dict[str, list] = {}
    for e in fks:
        out_fks.setdefault(e.src_table, []).append(e)
        in_fks.setdefault(e.dst_table, []).append(e)

    # ── Comp 5: confirmation surface — materialize RI-true rows (deterministic, same Comp-4 machinery
    # as the DDL spine: enums + curated pools + LLM-seeded entity values) so the lineup SHOWS sample data
    # and the value/de-canning quality is legible at /lineup, not just asserted by the gate. ───────────
    try:
        from aegir.ontology.chapter_tables import definitions_for_spine, entity_pools_for_spine
        from aegir.ontology.rows import materialize_rows, table_rows_as_records
        materialize_rows(spine, fks, definitions=definitions_for_spine(spine),
                         entity_pools=entity_pools_for_spine(spine))
    except Exception as e:                            # noqa: BLE001 — notes are valid without rows
        print(f"  [relational] row materialization skipped: {type(e).__name__}: {e}")
        table_rows_as_records = None  # type: ignore[assignment]

    # Per-anchor column-name entropy (de-canning, the principled metric — see semantic_layer_gate / Comp 4).
    _anchor_cnt: dict[str, Counter] = {}
    for st in spine:
        anc = (list(st.template.bfo_anchor_path) or ["(none)"])[-1]
        c = _anchor_cnt.setdefault(anc, Counter())
        for col in st.table.columns:
            if col.name != "id":
                c[col.name] += 1

    def _entropy(cnt: Counter) -> float:
        tot = sum(cnt.values())
        return -sum((v / tot) * math.log2(v / tot) for v in cnt.values() if v > 0) if tot else 0.0

    anchor_h = {a: _entropy(c) for a, c in _anchor_cnt.items()}
    _SCHEMAPILE_H_P10 = 2.585  # de-canning floor (SchemaPile p10); ≥ ⇒ real-DB-level vocabulary diversity

    # Cell-value classifier (mirrors scripts/check_value_semantics): placeholder / typed / domain.
    _PH = re.compile(r"^[A-Za-z][A-Za-z]*( [A-Za-z]+)* +\d{2,}$")
    _NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
    _ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}|$)")

    def _classify(v: str) -> str:
        s = (v or "").strip()
        if not s:
            return "empty"
        if _PH.match(s):
            return "placeholder"
        if _NUM.match(s) or _ISO.match(s) or s.lower() in ("true", "false"):
            return "typed"
        return "domain"

    def _rel_wl(table_name: str) -> str | None:
        tid = name2tid.get(table_name)
        return N.wl(f"relational/table/{_table_id(tid)}", tid) if tid else None

    out: list[N.Note] = []
    cat_tables: dict[str, list[str]] = {}
    for st in spine:
        t, cat, cols = st.template, st.family, st.table.columns
        rid = f"relational/table/{_table_id(t.template_id)}"
        cat_tables.setdefault(cat, []).append(rid)
        rowsmd = "\n".join(f"| `{c.name}` | {c.slot_type} | {c.slot_ref} |" for c in cols)
        ofk = [e for e in out_fks.get(st.table.name, []) if e.dst_table in name2tid]
        ifk = [e for e in in_fks.get(st.table.name, []) if e.src_table in name2tid]
        fk_md = ""
        if ofk:
            fk_md += "\n**Foreign keys** — follow → to the referenced table:\n\n" + "\n".join(
                f"- `{e.src_col}` → {_rel_wl(e.dst_table)} · `{e.dst_col}`  _(via {e.via_slot})_"
                for e in ofk) + "\n"
        if ifk:
            fk_md += "\n**Referenced by** — ← these tables foreign-key into this one:\n\n" + "\n".join(
                f"- {_rel_wl(e.src_table)} · `{e.src_col}`  _(via {e.via_slot})_" for e in ifk) + "\n"
        if not ofk and not ifk:
            fk_md = "\n_No cross-family foreign keys (standalone table)._\n"

        # ── Comp 5: verbalization + sample rows + quality badge (the confirmation surface) ──
        anchor = (list(t.bfo_anchor_path) or ["(none)"])[-1]
        frames = t.frames()
        vb_md = ""
        if frames:
            extra = f"  _(+{len(frames) - 1} more frames, sampled per chapter)_" if len(frames) > 1 else ""
            vb_md = f"\n**Verbalization.** _{frames[0]}_{extra}\n"

        pk_names = {c.name for c in cols if c.slot_ref == "__pk__"}
        fk_names = {e.src_col for e in ofk}
        sample_md, quality = "", {}
        if table_rows_as_records is not None and st.table.rows:
            records = table_rows_as_records(st)
            content = [v for rec in records for k, v in rec.items()
                       if k not in pk_names and k not in fk_names]
            n = len(content) or 1
            cls = Counter(_classify(v) for v in content)
            domain_frac, ph_frac = cls["domain"] / n, cls["placeholder"] / n
            h = anchor_h.get(anchor, 0.0)
            dot = "🟢" if (domain_frac >= 0.40 and ph_frac <= 0.30) else ("🟡" if domain_frac >= 0.20 else "🔴")
            hmark = "✓" if h >= _SCHEMAPILE_H_P10 else "·"
            quality = {"domain_fraction": round(domain_frac, 3), "placeholder_ratio": round(ph_frac, 3),
                       "n_verbalization_frames": len(frames), "anchor": anchor,
                       "decanning_h_colset": round(h, 2), "decanning_pass": bool(h >= _SCHEMAPILE_H_P10)}
            shown = list(records[0].keys())
            hdr = "| " + " | ".join(f"`{k}`" for k in shown) + " |"
            sep = "|" + "|".join("---" for _ in shown) + "|"
            body_rows = "\n".join("| " + " | ".join(str(rec[k]) for k in shown) + " |" for rec in records[:3])
            sample_md = (f"\n**Sample rows** (RI-true, deterministic):\n\n{hdr}\n{sep}\n{body_rows}\n"
                         f"\n**Quality** — {dot} domain {domain_frac:.2f} · placeholder {ph_frac:.2f} · "
                         f"{len(frames)} verbalization frame{'s' if len(frames) != 1 else ''} · "
                         f"de-canning H={h:.2f} {hmark} (anchor `{anchor}`)\n")

        body = (
            f"**Realizes term.** {N.wl(f'ontology/term/{t.template_id}', t.template_id)}  "
            f"(the ontology↔DDL pivot — DeepOnto semantics ↔ polyglot SQL syntax)\n"
            f"{vb_md}\n"
            f"**Columns** ({len(cols)}).\n\n"
            f"| column | owl/sql type | slot |\n|---|---|---|\n{rowsmd}\n"
            f"{sample_md}"
            f"{fk_md}"
        )
        out.append(N.Note(
            id=rid, title=st.table.name, kind="relational-table", data_product="relational",
            root="archive",
            body=body, frontmatter={
                "category": cat, "realizes": t.template_id, "table_name": st.table.name,
                "columns": [{"name": c.name, "type": c.slot_type, "slot": c.slot_ref} for c in cols],
                "not_null": sorted(getattr(st, "not_null", set()) or set()),
                "quality": quality,
                "foreign_keys": [{"column": e.src_col, "references_table": name2tid.get(e.dst_table),
                                  "references_column": e.dst_col, "via": e.via_slot} for e in ofk],
                "referenced_by": [{"table": name2tid.get(e.src_table), "column": e.src_col,
                                   "via": e.via_slot} for e in ifk]}))

    for cat, rids in sorted(cat_tables.items()):
        body = f"Relational tables in category **{cat}** — {len(rids)}.\n\n" + \
               "\n".join(f"- {N.wl(rid, rid.split('/')[-1])}" for rid in sorted(rids))
        out.append(N.Note(id=f"relational/category/{cat}", title=f"{cat} (tables)",
                          kind="relational-category", data_product="relational", body=body,
                          frontmatter={"category": cat, "n_tables": len(rids)}))
    print(f"  relational: {len(spine)} tables, {len(fks)} cross-family FKs (U2 FK-following edges)")
    return out


# ── content Data Product (on-disk corpus rows, if present) ───────────────────
def _windows(full: str, w: int = 12000) -> list:
    """Split text into ~``w``-char windows at LINE boundaries, never breaking inside a ``` code fence — so a
    chapter's serialized JSON table payload stays whole for the panel renderer."""
    out: list = []
    cur: list = []
    n = 0
    fence = False
    for ln in full.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
        cur.append(ln)
        n += len(ln) + 1
        if n >= w and not fence:
            out.append("\n".join(cur))
            cur, n = [], 0
    if cur:
        out.append("\n".join(cur))
    return out or [""]


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
        reg = r.get("_register", "natural")
        full = r.get("response_text") or ""
        wins = _windows(full, 12000)
        n = len(wins)
        body0 = f"*Register: **{reg}***\n\n{head}\n\n{wins[0]}"
        if n > 1:   # the next 12K window opens in its own panel (panel-trail wikilink — NOT bold-wrapped:
            # the UI's inline regex alternates [[..]] | **..**, so bold around a wikilink swallows it)
            body0 += f"\n\n→ {N.wl(f'content/chapter/{cid}__w2', f'continue — window 2 of {n}')}"
        out.append(N.Note(
            id=f"content/chapter/{cid}", title=_chapter_title(cid, reg) + (f" (1/{n})" if n > 1 else ""),
            kind="content-chapter", data_product="content", body=body0 + "\n", frontmatter={
                "category": r.get("family"), "model": r.get("model"), "ablation": r.get("ablation"),
                "register": reg, "target_topic_id": topic, "template_ids": tids, "windows": n}))
        for k in range(1, n):   # continuation windows — each its own panel, prev/next linked
            prev = f"content/chapter/{cid}" if k == 1 else f"content/chapter/{cid}__w{k}"
            # the lineup always opens to the RIGHT (panel-trail) — every window link is → (no back arrow),
            # disambiguated by window number rather than previous/next direction
            nav = f"→ {N.wl(prev, f'window {k} of {n}')}"
            if k + 1 < n:
                nav += f"  ·  → {N.wl(f'content/chapter/{cid}__w{k + 2}', f'window {k + 2} of {n}')}"
            out.append(N.Note(
                id=f"content/chapter/{cid}__w{k + 1}", title=f"{_chapter_title(cid, reg)} ({k + 1}/{n})",
                kind="content-chapter", data_product="content", body=f"{nav}\n\n{wins[k]}\n",
                frontmatter={"register": reg, "window": k + 1, "of": n}))
    from collections import Counter
    by_reg = Counter(r.get("_register", "natural") for r in recs)
    breakdown = " · ".join(f"{n} {reg}" for reg, n in sorted(by_reg.items())) or "—"
    body = (f"**Corpus** — {len(chapter_ids)} chapters ({breakdown}). Both registers mixed; "
            f"see {N.wl('lens/register', 'the register lens')}.\n\n"
            + "\n".join(f"- {N.wl(f'content/chapter/{c}', f'chapter {c}')}" for c in chapter_ids)
            + f"\n\nFinePDFs topics: {N.wl('topic/index')}\n")
    out.append(N.Note(id="content/index", title="Corpus", kind="content-index",
                      data_product="content", body=body,
                      frontmatter={"n_chapters": len(chapter_ids), "registers": dict(by_reg)}))
    return out


def project_registers(recs: list[dict]) -> list[N.Note]:
    """The register lens — chapters grouped by register: the **natural** canonical deliverable vs the
    **semantic** ontology-discourse register. Both are intentional, training-valuable surfaces of the same
    ontology+FinePDFs input (semantic prose elucidates the ontology's materialization)."""
    by_reg: dict[str, list[str]] = {}
    for i, r in enumerate(recs):
        by_reg.setdefault(r.get("_register", "natural"), []).append(_chapter_id(r, i))
    desc = {"natural": "natural physical names + topical practitioner prose (canonical deliverable)",
            "semantic": "ontology-discourse register — elucidates how/why the ontology materializes"}
    rows = ["| register | chapters | surface |", "|---|---|---|"]
    links = []
    for reg in sorted(by_reg):
        cids = by_reg[reg]
        rows.append(f"| {N.wl('lens/register', reg)} | {len(cids)} | {desc.get(reg, '')} |")
        links.append(f"\n**{reg}** ({len(cids)}): "
                     + " · ".join(N.wl(f"content/chapter/{c}", c) for c in cids[:12])
                     + (f" …(+{len(cids) - 12})" if len(cids) > 12 else ""))
    body = ("**Register × Corpus.** The content Data Product in two registers from the same ontology+FinePDFs "
            "input — both intentional training surfaces.\n\n" + "\n".join(rows) + "\n" + "\n".join(links))
    return [N.Note(id="lens/register", title="Register × Corpus", kind="lens", data_product="content",
                   frontmatter={"lens": "register", "registers": {k: len(v) for k, v in by_reg.items()}},
                   body=body)]


# ── topics (coverage rows, if present) — bridge content↔ontology ─────────────
def project_topics(recs: list[dict], topic_chapters: dict[int, list[str]] | None = None,
                   topic_colls: dict[int, list[str]] | None = None) -> list[N.Note]:
    topic_chapters = topic_chapters or {}
    topic_colls = topic_colls or {}
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
        cls = topic_colls.get(tid, [])
        if cls:
            edges.append(f"**Threads** {len(cls)} collection(s): "
                         + ", ".join(N.wl(c, c.split('-')[-1]) for c in cls[:8]) + (" …" if len(cls) > 8 else ""))
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


# ── collections (topic-grounded bundles; the rows the landing pivots) ────────
def _idlist(v) -> list:
    """A parquet list field that may arrive as a real list or a stringified one."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str) and v.strip():
        try:
            import ast
            return list(ast.literal_eval(v))
        except Exception:  # noqa: BLE001
            return []
    return []


def project_collections(recs: list[dict], coverage: list[dict]) -> tuple[list[N.Note], dict]:
    """De-flattened, many-to-many collections — the unit the landing pivots. A collection's
    DOCUMENTS (chapters) are the hub: it relates to MANY topics (target ∪ style across its
    chapters) and many terms/tables, and each terminal recurs across collections. Returns
    (notes, maps); maps drive the lens pivots and the transpose backlinks."""
    by_coll: dict[int, list[tuple[int, dict]]] = {}
    for i, r in enumerate(recs):
        t = r.get("target_topic_id")
        if t is not None:
            by_coll.setdefault(int(t), []).append((i, r))
    cov = {int(c["topic_id"]): c for c in coverage}
    coll_terms: dict[str, list[str]] = {}
    coll_topics: dict[str, list[int]] = {}
    coll_chapters: dict[str, list[str]] = {}
    notes: list[N.Note] = []
    for tid in sorted(by_coll):
        chs = by_coll[tid]
        cid = f"collection/topic-{tid:03d}"
        cids = [_chapter_id(r, i) for i, r in chs]
        topics = sorted({tid} | {int(s) for _, r in chs for s in _idlist(r.get("style_topic_ids"))})
        terms = sorted({str(x) for _, r in chs for x in _idlist(r.get("template_ids"))})
        coll_terms[cid], coll_topics[cid], coll_chapters[cid] = terms, topics, cids
        gist = re.sub(r"\s+", " ", (cov.get(tid, {}).get("topic_repr_text") or "")).strip()[:280]
        body = (
            f"**Topic-grounded collection** — anchored at {N.wl(f'topic/{tid}', f'topic {tid}')}, "
            f"drawing on **{len(topics)} topics** (target + style · many-to-many), "
            f"**{len(terms)} terms**, **{len(cids)} chapters**.\n\n"
            + (f"> {gist} …\n\n" if gist else "")
            + "**Documents.** " + (" · ".join(N.wl(f"content/chapter/{c}", _chapter_title(c, r.get("_register")))
                                              for (_, r), c in zip(chs[:12], cids[:12])) or "—") + "\n\n"
            + "**Topics.** " + (" · ".join(N.wl(f"topic/{t}", f"topic {t}") for t in topics[:18]) or "—") + "\n\n"
            + "**Realizes terms.** " + (" · ".join(N.wl(f"ontology/term/{x}", x) for x in terms[:24]) or "—") + "\n\n"
            + "**Underlying tables.** " + (" · ".join(N.wl(f"relational/table/{_table_id(x)}", x) for x in terms[:24]) or "—") + "\n")
        notes.append(N.Note(id=cid, title=f"collection · topic {tid}", kind="collection",
                            data_product="content", body=body, frontmatter={
                                "topic_id": tid, "n_topics": len(topics), "n_terms": len(terms),
                                "n_chapters": len(cids), "topics": topics}))
    topic_colls: dict[int, list[str]] = {}
    term_colls: dict[str, list[str]] = {}
    for cid, tps in coll_topics.items():
        for t in tps:
            topic_colls.setdefault(t, []).append(cid)
    for cid, tms in coll_terms.items():
        for t in tms:
            term_colls.setdefault(t, []).append(cid)
    idx = (f"**Collections** — {len(by_coll)} topic-grounded bundles (every topic and base table "
           "recurs across collections — the graph is many-to-many).\n\n"
           + "\n".join(f"- {N.wl(f'collection/topic-{t:03d}', f'topic {t}')} "
                       f"({len(coll_chapters[f'collection/topic-{t:03d}'])} ch · "
                       f"{len(coll_terms[f'collection/topic-{t:03d}'])} terms)" for t in sorted(by_coll)))
    notes.append(N.Note(id="collection/index", title="Collections", kind="collection-index",
                        data_product="content", body=idx, frontmatter={"n_collections": len(by_coll)}))
    return notes, {"collections": sorted(by_coll), "coll_terms": coll_terms,
                   "coll_topics": coll_topics, "topic_colls": topic_colls, "term_colls": term_colls}


# ── lenses (the landing pivot: collections × the lens-selected axis) ──────────
def _relational_spine(rows):
    """(tid→table_name, FKEdge list) for the Schema chord's FK-spanning substrate — mirrors the
    spine project_relational builds. Guarded + lazy: the chord is optional enrichment, so any
    failure yields ({}, []) rather than breaking the build."""
    from aegir.ontology.ddl import cross_family_fks, template_to_table
    import aegir.ontology as _onto
    from aegir.ontology.complex import FamilyComplex
    spine = []
    for cat, t in rows:
        try:
            spine.append(template_to_table(t, cat))
        except Exception:  # noqa: BLE001
            pass
    tid_table = {st.template.template_id: st.table.name for st in spine}
    try:
        fc = FamilyComplex.from_json(Path(_onto.__file__).resolve().parent / "family_complex.json")
        fks, _ = cross_family_fks(spine, fc)
    except Exception:  # noqa: BLE001
        fks = []
    return tid_table, fks


def project_lenses(categories: list[str], has_content: bool, has_topics: bool,
                   maps: dict | None = None) -> list[N.Note]:
    maps = maps or {}
    colls = maps.get("collections")
    # terms (default): collections × realized terms — the grounding pivot
    if colls:
        rows = ["| collection | realizes terms |", "|---|---|"]
        for tid in colls:
            cid = f"collection/topic-{tid:03d}"
            tms = maps["coll_terms"][cid]
            cell = " · ".join(N.wl(f"ontology/term/{x}", x) for x in tms[:6]) + (f" …(+{len(tms) - 6})" if len(tms) > 6 else "")
            rows.append(f"| {N.wl(cid, f'topic {tid}')} | {cell or '—'} |")
        terms_body = ("**Lexicon × Collections.** Each collection (a topic-grounded bundle) and the "
                      "ontology terms it realizes — the grounding made visible. Click a term to pivot "
                      "to *its* collections; a collection for its full bundle.\n\n" + "\n".join(rows))
    else:
        terms_body = ("**Lexicon** `" + LEXICON + "`. Browse by category:\n\n"
                      + "\n".join(f"- {N.wl(f'ontology/category/{c}', c)}" for c in categories))
    terms = N.Note(id="lens/terms", title="Lexicon × Collections" if colls else "Lexicon", kind="lens",
                   data_product="ontology", frontmatter={"lens": "terms", "lexicon": LEXICON}, body=terms_body)
    # schema: collections' footprint organized around category (tables shared across collections)
    schema = N.Note(id="lens/schema", title="Schema × Collections" if colls else "Schema", kind="lens",
                    data_product="relational", frontmatter={"lens": "schema", "lexicon": LEXICON},
                    body=("**Schema × Collections.** The relational footprint by category — a base "
                          "table is shared across the collections whose chapters embed views over it "
                          "(many-to-many). By category:\n\n"
                          + "\n".join(f"- {N.wl(f'relational/category/{c}', c)}" for c in categories)))
    # content: oriented around topics → topic → its collections (the densest cross-axis)
    tc = (maps or {}).get("topic_colls")
    if colls and tc:
        rows = ["| topic | collections it threads |", "|---|---|"]
        for t in sorted(tc, key=lambda k: -len(tc[k]))[:80]:
            cs = tc[t]
            cell = " · ".join(N.wl(c, c.split("-")[-1]) for c in cs[:8]) + (f" …(+{len(cs) - 8})" if len(cs) > 8 else "")
            rows.append(f"| {N.wl(f'topic/{t}', f'topic {t}')} | {cell} |")
        content_body = ("**Content × Topics.** FinePDFs topics and the collections they thread through "
                        "(target + style — many-to-many; a topic spans many collections). Click a topic "
                        "for its gist, a collection for its bundle.\n\n" + "\n".join(rows)
                        + f"\n\nAll: {N.wl('collection/index', 'collections')} · {N.wl('topic/index', 'topics')}")
    else:
        content_body = ("**Content.** The corpus + FinePDFs topics.\n\n"
                        + (f"- {N.wl('content/index', 'the corpus chapters')}\n- {N.wl('topic/index', 'the FinePDFs topics')}"
                           if has_content or has_topics else "No corpus/topics projected yet."))
    content = N.Note(id="lens/content", title="Content × Topics" if colls else "Content", kind="lens",
                     data_product="content", frontmatter={"lens": "content"}, body=content_body)

    # The lens chords render live via the bokeh server (aegir.viz.lineup_app), embedded by the React
    # <PanelView> — no chord is baked into the note frontmatter anymore.
    return [terms, schema, content]


def project_training() -> list[N.Note]:
    """Training/experiment lineup notes — launchers for live HoloViews viz (a note's ``viz_app``
    frontmatter names the bokeh-server app the panel mounts; data reads live from ``outputs/runs``).
    Extensible: more Training entries (gate panels, ablation overlays, …) land here as requirements
    materialize — see docs/current/src/roadmap/leaderboard_observatory.md."""
    sweeps = N.Note(
        id="training/sweeps", title="Sweeps", kind="training", data_product="training",
        frontmatter={"viz_app": "sweeps_app"},
        body=("**Sweeps.** Parallel-coordinates over training runs (live HoloViews) — each run is a "
              "line across hyperparam → outcome axes (model size · params · lr · epochs · macro/micro "
              "F1 · val loss), colored by best macro-F1. Read the Pareto front, clusters, and the "
              "effect of each knob at a glance. Reads runs live from `outputs/runs`; per-run detail "
              "lives on the Leaderboards page."))
    reward = N.Note(
        id="training/reward", title="Reward", kind="training", data_product="training",
        frontmatter={"viz_app": "reward_app"},
        body=("**Reward.** Live GRPO/RLVR reward dynamics — the verifier composite **R** (mean±std band, "
              "where the band *is* the variance collapse-canary), the **R_A pass-rate**, and the z-scored "
              "advantage, vs GRPO iteration. The health monitor for the RLVR loop: reward climbing, "
              "variance not collapsing, pass-rate rising. Reads the run's GRPO `metrics_jsonl` live from "
              "`cfg.p5.output_dir`."))
    provenance = N.Note(
        id="training/provenance", title="Provenance", kind="provenance", data_product="training",
        frontmatter={"ego_focal": None},
        body=("**Provenance — instance-level lineage, walkable.** The pipeline's lineage read live from the "
              "Atlas (`aegir_hx`) graph as a ReactFlow **ego-graph**: each panel shows ONE node's first-order "
              "neighborhood; **click a node to open its own neighborhood in a new panel** — walk the lineage "
              "in true lineup fashion. Versioned artifacts (Family · Topic · Template · Chapter · Column · "
              "Dataset · Run · Job) and their derivation edges (SELECTED · PRODUCED · SEEDED_BY · "
              "RE_GROUNDS_TO …). Next: per-edge gate-verdict overlays (R-pass · HermiT · coverage · "
              "downstream-eval-lift). See docs/current/src/roadmap/provenance.md."))
    return [sweeps, reward, provenance]


def project_metrics() -> list[N.Note]:
    """Training → Metrics: the curated catalog of the project's quantitative controls (gates · measures · KPIs)
    as a navigable tree (root → category → metric, panel-trail). From ``metrics_catalog.METRICS``."""
    from aegir.lineup.metrics_catalog import METRICS
    out: list[N.Note] = []
    total = sum(len(c["metrics"]) for c in METRICS)
    cat_lines = []
    for c in METRICS:
        cid = f"training/metrics/{c['slug']}"
        cat_lines.append(f"- {N.wl(cid, c['title'])} — {len(c['metrics'])} · {c['blurb'][:80]}…")
    out.append(N.Note(
        id="training/metrics", title="Metrics", kind="training", data_product="training",
        body=("**Metrics — the quantitative controls.** The gates, measures, and KPIs that steer the three "
              "coupled products (ontology · corpus · model) toward higher quality and drive model fine-tuning "
              f"+ H-Net+RWKV training. **{total} metrics across {len(METRICS)} categories** — open a category "
              "for its metrics; a metric for its definition · formula · gate · role · source.\n\n"
              + "\n".join(cat_lines))))
    for c in METRICS:
        cslug = c["slug"]
        m_lines = []
        for m in c["metrics"]:
            mid = f"training/metrics/{cslug}/{m['slug']}"
            g = f" — gate **{m['gate']}**" if m.get("gate") and m["gate"] != "—" else ""
            m_lines.append(f"- {N.wl(mid, m['name'])}{g}")
        cat_fm: dict = {}
        live_block = ""
        if cslug == "relational-shape":
            live_rs = S.relational_shape()
            if live_rs:
                cat_fm = {"relational_shape": live_rs}
                emd = live_rs.get("shape_emd")
                live_block = (f"\n\n**LIVE** (spine `{live_rs['spine_run']}`, {live_rs['n_tables']} tables: "
                              f"{live_rs['strata']['base']} base + {live_rs['strata']['view']} view): "
                              f"median **{live_rs['cols_median']}** · p90 {live_rs['cols_p90']} · "
                              f"p99 {live_rs['cols_p99']} · max {live_rs['cols_max']} · "
                              f"wide(≥20) {live_rs['wide_rate']:.1%} · "
                              f"EMD vs SchemaPile: {emd if emd is not None else 'pending #139 norms'}\n")
        if cslug == "ontology-rigor":
            live = S.ontology_metrology()
            if live:
                cat_fm = {"ontology_quality": live}
                ch = live["oquare_characteristics"]
                live_block = (
                    f"\n\n**Live** (`corpora/ontology/sdg-ontology.owl` · {live['n_domain_classes']} classes · "
                    f"consistent={live['consistent']}): OQuaRE **{live['oquare_aggregate']}/5** "
                    f"{'🟢 GREEN' if live['oquare_green'] else '🔴 RED'} · bfo-grounded **{live['bfo_grounded']:.0%}** · "
                    f"def-annotation **{live['def_annotation_coverage']:.0%}** · def-completeness "
                    f"**{live['definitional_completeness']:.1%}** · realizable **{live['realizable_machinery']}** · "
                    f"AR **{live['ar']:.3f}** ({live['n_datatype_properties']} dataprops)\n\n"
                    f"_characteristics (1-5)_ — Structural {ch['Structural']} · FunctionalAdequacy "
                    f"{ch['FunctionalAdequacy']} · Reliability {ch['Reliability']} · Operability {ch['Operability']} · "
                    f"Maintainability {ch['Maintainability']} · Transferability {ch['Transferability']}\n\n"
                    f"_OntoClean (un-gameable)_ — taxonomic-cleanliness **{live['taxonomic_cleanliness']}** · "
                    f"violations {live['ontoclean_violations']} · cycles {live['subsumption_cycles']} · "
                    f"sibling-disjoint {live['sibling_disjointness']}")
        out.append(N.Note(
            id=f"training/metrics/{cslug}", title=c["title"], kind="training", data_product="training",
            frontmatter=cat_fm,
            body=(f"{N.wl('training/metrics', '← all metrics')}\n\n**{c['title']}.** {c['blurb']}"
                  + live_block + "\n\n" + "\n".join(m_lines))))
        for m in c["metrics"]:
            g = f"\n- **gate / threshold:** {m['gate']}" if m.get("gate") and m["gate"] != "—" else ""
            out.append(N.Note(
                id=f"training/metrics/{cslug}/{m['slug']}", title=m["name"], kind="metric",
                data_product="training",
                body=(f"{N.wl(f'training/metrics/{cslug}', '← ' + c['title'])}\n\n"
                      f"- **definition / formula:** {m['formula']}{g}\n"
                      f"- **role:** {m['role']}\n- **defined in:** `{m['src']}`")))
    return out


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
    # template → aligned topic_ids (coverage top-template alignment — the rich `top_templates` list, ~5/topic).
    # Built FIRST so template-driven (refinement) chapters get an ORGANIC target_topic_id from the templates they
    # realize; without it they carry no topic and the topic/collections axis can't grow as the refined corpus does.
    term_topics: dict[str, list[int]] = {}
    for r in coverage:
        if int(r["topic_id"]) < 0:            # BERTopic outlier/noise cluster (-1) — not a real topic; never assign
            continue
        tset: set[str] = set()
        if r.get("top_template_id"):
            tset.add(str(r["top_template_id"]))
        for e in (r.get("top_templates") or []):
            t = e.get("template_id") if isinstance(e, dict) else e
            if t:
                tset.add(str(t))
        for t in tset:
            term_topics.setdefault(t, []).append(int(r["topic_id"]))
    for r in corpus:                              # template-driven chapters → derive their topic THREAD from the
        cur = r.get("target_topic_id")            # templates' coverage alignment: a dominant ANCHOR topic plus the
        if cur is not None and int(cur) < 0:      # secondary alignments as STYLE topics — so a collection spans
            r["target_topic_id"] = cur = None     # many topics and a topic threads many collections (the m:n graph).
        if r.get("style_topic_ids"):              # scrub outlier (-1) style topics carried by older generations
            r["style_topic_ids"] = [s for s in _idlist(r["style_topic_ids"]) if int(s) >= 0]
        if cur is None:
            votes: dict[int, int] = {}
            for tid in (r.get("template_ids") or []):
                for tp in term_topics.get(str(tid), []):
                    votes[tp] = votes.get(tp, 0) + 1
            if votes:
                ranked = sorted(votes, key=lambda k: (-votes[k], k))
                r["target_topic_id"] = ranked[0]
                r["style_topic_ids"] = ranked[1:5]   # secondary template-topic alignments → the m:n thread
    term_chapters: dict[str, list[str]] = {}
    topic_chapters: dict[int, list[str]] = {}
    for i, r in enumerate(corpus):
        cid = _chapter_id(r, i)
        for tid in (r.get("template_ids") or []):
            term_chapters.setdefault(tid, []).append(cid)
        tp = r.get("target_topic_id")
        if tp is not None:
            topic_chapters.setdefault(int(tp), []).append(cid)

    broader, narrower = S.term_hierarchy()

    # Collections (the de-flattened many-to-many graph the landing pivots) + transpose maps.
    coll_notes: list[N.Note] = []
    maps: dict = {}
    if corpus and coverage:
        coll_notes, maps = project_collections(corpus, coverage)

    notes = (project_ontology(rows, term_chapters, term_topics, broader, narrower,
                              term_colls=maps.get("term_colls"))
             + project_relational(rows))
    print(f"  ontology+relational: {len(notes)} notes from {len(rows)} terms "
          f"in {len(categories)} categories (Lexicon {LEXICON!r})")
    n_edges = sum(len(v) for v in broader.values())
    if n_edges:
        print(f"  hierarchy: {n_edges} HermiT-verified subsumption edges over "
              f"{len(broader)} child terms (from catalog `broader` — the SoT)")
    else:
        print("  hierarchy: (no subsumption edges — run scripts/mediate_hierarchy.py [--all] then --promote)")

    if corpus:
        c = project_content(corpus)
        notes += c
        rl = project_registers(corpus)
        notes += rl
        print(f"  content: {len(c)} notes  ({crun})  +register lens {rl[0].frontmatter.get('registers')}")
    else:
        print("  content: (no on-disk corpus run — skipped; set AEGIR_CORPUS_RUN to project)")
    if coverage:
        tp = project_topics(coverage, topic_chapters, topic_colls=maps.get("topic_colls"))
        notes += tp
        print(f"  topics:  {len(tp)} notes   ({cov})")
    else:
        print("  topics:  (no on-disk coverage run — skipped; set AEGIR_COVERAGE_RUN to project)")

    if coll_notes:
        notes += coll_notes
        print(f"  collections: {len(coll_notes) - 1} topic-grounded bundles (many-to-many: "
              f"{len(maps.get('topic_colls', {}))} topics × {len(maps.get('term_colls', {}))} terms)")

    notes += project_lenses(categories, bool(corpus), bool(coverage), maps)
    notes += project_training()
    mt = project_metrics()
    notes += mt
    print(f"  training: 3 viz panels + metrics catalog ({len(mt)} notes)")

    sc = S.sdg_constructs()
    if sc:
        tbls, vws = sc["tables"], sc["views"]
        known = set(tbls)
        t_lines = [f"- {N.wl('relational/table/' + n, n)} — {len(e['columns'])} cols · "
                   f"{len(e['fks'])} FKs · pk {e.get('pk_kind') or e.get('pk') or '—'}"
                   for n, e in sorted(tbls.items())]
        notes.append(N.Note(
            id="relational/sdg-schema", title="SDG schema (generated)", kind="relational",
            data_product="relational",
            body=(f"**The generated relational product, verbatim** — {len(tbls)} unique tables "
                  f"+ {len(vws)} views across {sc['n_constructs']} constructs (`just metaflow` "
                  f"corpus). Table and column names are the artifacts themselves — no wrappers.\n\n"
                  + "\n".join(t_lines))))
        for n, e in tbls.items():
            fk_lines = [f"- `{fk.get('col')}` → " +
                        (N.wl("relational/table/" + fk.get("ref_table", ""), fk.get("ref_table", ""))
                         if fk.get("ref_table") in known else f"`{fk.get('ref_table')}`") +
                        f" · `{fk.get('ref_col')}`"
                        for fk in e["fks"]]
            prov = ", ".join(e["constructs"][:4]) + ("…" if len(e["constructs"]) > 4 else "")
            notes.append(N.Note(
                id=f"relational/table/{n}", title=n, kind="relational-table",
                data_product="relational",
                frontmatter={"pk": e.get("pk"), "pk_kind": e.get("pk_kind"),
                             "n_columns": len(e["columns"]), "constructs": len(e["constructs"])},
                links=[f"relational/table/{fk.get('ref_table')}" for fk in e["fks"]
                       if fk.get("ref_table") in known],
                body=(f"**`{n}`** — generated table (verbatim; pk kind "
                      f"**{e.get('pk_kind') or '—'}**, pk `{e.get('pk') or '—'}`).\n\n"
                      f"Columns: " + " · ".join(f"`{c}`" for c in e["columns"]) + "\n\n"
                      + ("**Foreign keys**\n" + "\n".join(fk_lines) + "\n\n" if fk_lines else "")
                      + f"_Constructs: {prov}_")))
        print(f"  relational(sdg): {len(tbls)} tables + {len(vws)} views VERBATIM "
              f"from {sc['n_constructs']} constructs", flush=True)

    gc = S.sdg_corpus()
    if gc:
        m = gc.get("metrics") or {}
        st = gc.get("structure") or m.get("structure") or {}
        cg = gc.get("congruence") or {}
        cg_line = ""
        if cg:
            nat, sem = cg.get("natural") or {}, cg.get("semantic") or {}
            cg_line = (f"\n- **Concept congruence** (input window → chapters): natural "
                       f"recovery {nat.get('top1_recovery_rate', '—')} / profile "
                       f"{nat.get('mean_profile_congruence', '—')} · semantic "
                       f"{sem.get('top1_recovery_rate', '—')} / {sem.get('mean_profile_congruence', '—')}")
        notes.append(N.Note(
            id="corpus/sdg", title="SDG corpus (live)", kind="corpus",
            data_product="corpus",
            frontmatter={"live": True, "passages": gc["passages_derived"],
                         "chapters": gc["chapters_natural"] + gc["chapters_semantic"],
                         "shape_emd": st.get("shape_emd")},
            body=(f"**The accreting SDG corpus** — `just metaflow` tops this up per input "
                  f"window (idempotent; content-hash passages + deriver-version stamps).\n\n"
                  f"- **{gc['passages_derived']} passages derived** → "
                  f"{gc['chapters_natural']} natural + {gc['chapters_semantic']} semantic chapters\n"
                  f"- Merged ontology: {st.get('n_elected', '—')} tables · "
                  f"{st.get('total_fks', '—')} FKs · {st.get('n_junctions', '—')} junctions · "
                  f"{st.get('n_lookups', '—')} lookups · **shape EMD {st.get('shape_emd', '—')}**\n"
                  f"- Payload: {(m.get('payload') or {}).get('tables_embedded', '—')} tables + "
                  f"{(m.get('payload') or {}).get('views_embedded', '—')} views embedded"
                  + cg_line +
                  f"\n\nSource: `{gc['root']}` (metrics.json · congruence.json · concept_graph.json)")))
        print(f"  corpus: sdg live note ({gc['passages_derived']} passages, "
              f"{gc['chapters_natural'] + gc['chapters_semantic']} chapters)")
        from aegir.lineup.zettel import run_zettels
        zs = run_zettels(Path(gc["root"]))
        for z in zs:
            w, st = z.get("window") or {}, z.get("state") or {}
            notes.append(N.Note(
                id=f"corpus/runs/{z['id']}", title=z["id"], kind="corpus-run",
                data_product="corpus",
                frontmatter={"prev": z.get("prev"), "at": z.get("at"),
                             "deriver": (z.get("versions") or {}).get("deriver"),
                             "commit": (z.get("versions") or {}).get("commit")},
                links=[f"corpus/runs/{z['prev']}"] if z.get("prev") else [],
                body=(f"**Run-zettel {z['id']}** (metaflow {z.get('metaflow_run_id', '?')})\n\n"
                      f"- window: cursor {w.get('harvest_cursor')} · "
                      f"{w.get('passages_fresh', 0)} fresh + {w.get('passages_cached', 0)} cached\n"
                      f"- versions: deriver `{(z.get('versions') or {}).get('deriver')}` @ "
                      f"`{(z.get('versions') or {}).get('commit')}`\n"
                      f"- state: {st.get('n_chapters')} chapters · "
                      f"EMD {(st.get('structure') or {}).get('shape_emd')} · "
                      f"congruence {json.dumps(st.get('congruence'), default=str)[:120]}\n"
                      + (f"- prev: {N.wl('corpus/runs/' + z['prev'], z['prev'])}" if z.get("prev") else "- chain origin"))))
        if zs:
            print(f"  corpus: {len(zs)} run-zettels (chain head {zs[-1]['id']})")

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
