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


def _fill_slot(m: "re.Match") -> str:
    """{RightsExerciseProcess} → 'rights exercise process' — slots FILL as their de-camelized
    names; deleting them truncated every braced verbalization to a subjectless fragment
    ('is a process that has participant some' — RH 2026-07-21)."""
    name = m.group(0).strip("{}").split(":")[0]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def _verbal(t: CatalogTemplate) -> str:
    v = re.sub(r"\{[^}]+\}", _fill_slot, t.verbal_template or "")
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
                     term_colls: dict[str, list[str]] | None = None,
                     term_items: dict[str, list[str]] | None = None) -> list[N.Note]:
    term_chapters = term_chapters or {}
    term_topics = term_topics or {}
    term_broader = term_broader or {}
    term_narrower = term_narrower or {}
    term_colls = term_colls or {}
    term_items = term_items or {}
    cats: dict[str, list[str]] = {}
    cat_meta: dict[str, dict] = {}
    anchors: dict[str, dict] = {}
    out: list[N.Note] = []

    for cat, t in rows:
        term_id = f"ontology/term/{t.template_id}"
        cat_id = f"ontology/category/{cat}"
        cats.setdefault(cat, []).append(term_id)
        prov = t.provenance or {}
        cm = cat_meta.setdefault(cat, {"tier": None, "grounds": {}})
        cm["tier"] = cm["tier"] or prov.get("tier")
        g = S.relational_category(t)
        cm["grounds"][g] = cm["grounds"].get(g, 0) + 1
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
        itm = term_items.get(t.template_id, [])
        if itm:
            body += ("\n**Aligned items** (inverted topic layer — FinePDFs windows that "
                     "unambiguously bind here): " + str(len(itm)) + " — "
                     + ", ".join(N.wl(i, i.split("/")[-1][:8]) for i in itm[:10])
                     + (" …" if len(itm) > 10 else "") + "\n")
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
                "tier": prov.get("tier"),
                "domain": (prov.get("domain") or {}).get("label") if isinstance(prov.get("domain"), dict) else prov.get("domain"),
                "grounds_ddl": S.relational_category(t),
                "broader": bro, "narrower": nar,
                "skos": skos, "retrieval_text": rtext,
                "provenance": dict(t.provenance or {})}))

    for cat, term_ids in sorted(cats.items()):
        qn = category_qn(cat)
        cm = cat_meta.get(cat) or {}
        grounds = " · ".join(f"`{g}`×{n}" for g, n in sorted((cm.get("grounds") or {}).items(),
                                                             key=lambda kv: -kv[1]))
        head = (f"**Axiom pattern** `{cat}` (tier `{cm.get('tier')}`) — {len(term_ids)} terms. "
                f"Grounds: {grounds}." if cm.get("tier") else
                f"**Intermediate domain classes** (CTA subsumers, defined by the agent-mediated "
                f"loop) — {len(term_ids)} terms.")
        body = (
            f"{head} Part of the {N.wl('lens/terms', 'Lexicon')} `{qn}`.\n\n"
            + "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(term_ids))
        )
        out.append(N.Note(id=f"ontology/category/{cat}", title=cat, kind="ontology-category",
                          data_product="ontology", body=body, frontmatter={
                              "qualified_name": qn, "parent": None, "children": [],
                              "n_terms": len(term_ids), "tier": cm.get("tier"),
                              "grounds": cm.get("grounds") or {}}))

    for aid, a in sorted(anchors.items()):
        body = f"BFO/CCO anchor **{a['label']}** — {len(a['terms'])} terms anchored here (siblings).\n\n"
        if a["parent"]:
            body += f"**Broader.** {N.wl(a['parent'])}\n\n"
        body += "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(a["terms"]))
        out.append(N.Note(id=aid, title=a["label"], kind="ontology-anchor",
                          data_product="ontology", body=body,
                          frontmatter={"anchor": a["label"], "n_terms": len(a["terms"])}))
    return out


# ── ITEM notes — the FedWiki granularity (the inverted topic layer's lineage) ────
def project_items(assoc: dict, live_terms: set) -> "tuple[list[N.Note], dict[str, list[str]]]":
    """Project the inverted topic layer's association records as ITEM notes (trunk).

    An item is a passage window (content-addressed, span-faithful); its note IS the
    lineage record made walkable: the window text, the adjudication (topic · score ·
    hierarchical margin · collection state · encoder — RH rulings c/d/g), the topic as
    a live term wikilink, and the containing document. Only ASSIGNED items get notes;
    the ambiguous mass stays in the run report (it is the rigor loop's food, not a
    browse surface). Returns (notes, term→item-ids) so term notes become the hubs."""
    report = assoc.get("report") or {}
    assigned = [r for r in assoc["records"] if r.get("assigned")]
    docs_dir = S.REPO / "build" / "domain_harvest" / "docs"
    doc_cache: dict[str, str] = {}
    out: list[N.Note] = []
    term_items: dict[str, list[str]] = {}
    by_topic: dict[str, list[str]] = {}
    seen: set[str] = set()
    for r in assigned:
        iid = f"item/{r['passage_hash'][:16]}"
        if iid in seen:
            continue
        seen.add(iid)
        src = r.get("source") or ""
        if src not in doc_cache:
            p = docs_dir / src
            doc_cache[src] = p.read_text(errors="ignore") if (src and p.exists()) else ""
        s, e = ((r.get("item_span") or []) + [0, 0])[:2]
        text = doc_cache[src][s:e] if doc_cache[src] else ""
        code = r.get("topic_code") or ""
        is_term = code in live_terms
        topic_wl = (N.wl(f"ontology/term/{code}", code) if is_term
                    else f"`{code}` {r.get('topic_label', '')} (domain roll-up)")
        body = (
            f"**Item** `{r['passage_hash'][:16]}` — a passage window "
            f"(chars {s}–{e} of doc `{(r.get('doc_hash') or '')[:16]}`, `{src}`).\n\n"
            + (f"> {text[:900]}{'…' if len(text) > 900 else ''}\n\n" if text else "")
            + f"**Topic.** {topic_wl}\n\n"
            f"**Adjudication.** score {r.get('score')} · margin_h **{r.get('rel_margin_h')}** "
            f"(flat {r.get('rel_margin')}) vs competitor `{r.get('competitor_code') or '—'}` · "
            f"τ {r.get('tau')}\n\n"
            f"**Lineage.** collection `{r.get('collection')}@{r.get('collection_sha')}` · "
            f"encoder `{r.get('encoder')}` · projection `{r.get('projection', '-')}` · "
            f"{r.get('n_tokens')} tokens\n\n"
            f"All items: {N.wl('item/index', 'the item index')}")
        out.append(N.Note(
            id=iid, title=f"item {r['passage_hash'][:12]}", kind="item",
            data_product="content", root="scratch", body=body,
            frontmatter={"doc_hash": r.get("doc_hash"), "source": src,
                         "span": r.get("item_span"), "topic_code": code,
                         "topic_is_term": is_term, "score": r.get("score"),
                         "rel_margin_h": r.get("rel_margin_h"),
                         "collection_sha": r.get("collection_sha"),
                         "encoder": r.get("encoder")}))
        by_topic.setdefault(code, []).append(iid)
        if is_term:
            term_items.setdefault(code, []).append(iid)

    lines = []
    for code, iids in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
        head = (N.wl(f"ontology/term/{code}", code) if code in live_terms
                else f"`{code}` (domain)")
        lines.append(f"- {head} — {len(iids)}: "
                     + " · ".join(N.wl(i, i.split('/')[-1][:8]) for i in iids[:10])
                     + (" …" if len(iids) > 10 else ""))
    out.append(N.Note(
        id="item/index", title="Items × Topics (inverted layer)", kind="item-index",
        data_product="content", root="scratch",
        frontmatter={"collection_sha": report.get("collection_sha"),
                     "n_items": report.get("n"), "n_assigned": report.get("n_assigned"),
                     "alignment_rate": report.get("alignment_rate"),
                     "topics_hit": report.get("topics_hit")},
        body=(f"**Items × Topics** — the inverted topic layer's lineage, walkable. "
              f"{report.get('n_assigned')}/{report.get('n')} items aligned "
              f"(rate {report.get('alignment_rate')}) across {report.get('topics_hit')} topics; "
              f"registry `{report.get('collection')}@{report.get('collection_sha')}`, "
              f"τ {report.get('tau')}. One passage window → ONE topic (hierarchical margin) "
              f"or none — the unassigned mass drives the definitional-rigor loop.\n\n"
              + "\n".join(lines))))
    return out, term_items


def project_retired_ontology(rows: list[tuple[str, CatalogTemplate]],
                             term_chapters: dict[str, list[str]] | None = None,
                             term_topics: dict[str, list[int]] | None = None,
                             term_colls: dict[str, list[str]] | None = None,
                             cited: set | None = None,
                             era_tables: dict[str, str] | None = None) -> list[N.Note]:
    """The superseded catalog generations, ERA-ALIGNED into the roots (RH 2026-07-06):
    ids the RELEASED corpus/coverage cite are the release's lexicon → ``current``
    citizens (that corpus IS the current release, and these are its terms); the
    recovered-but-uncited remainder → ``archive`` tombstones. ``era_tables`` maps a
    template to its RELEASED spine table (corpora/ddl) so the release kasten pivots
    term↔table within the current root."""
    term_chapters = term_chapters or {}
    term_topics = term_topics or {}
    term_colls = term_colls or {}
    cited = cited or set()
    era_tables = era_tables or {}
    out: list[N.Note] = []
    fams: dict[str, list[str]] = {}
    fam_cited: dict[str, int] = {}
    for fam, t in rows:
        term_id = f"ontology/term/{t.template_id}"
        fams.setdefault(fam, []).append(term_id)
        is_cited = t.template_id in cited
        fam_cited[fam] = fam_cited.get(fam, 0) + (1 if is_cited else 0)
        slots = ", ".join(f"`{s}` ({owl})" for s, owl in (t.slot_types or {}).items())
        if is_cited:
            banner = (f"**Release-era term** (family `{fam}`) — part of the lexicon the "
                      f"RELEASED corpus cites (the current release). Superseded on trunk "
                      f"(`scratch`) by the content-first derive.")
        else:
            banner = (f"**RETIRED** — a superseded catalog generation (family `{fam}`), "
                      f"cited by nothing current; an archive tombstone.")
        body = (
            f"{banner}\n\n"
            f"**Verbalization.** {_verbal(t)}\n\n"
            f"**Axiom (Manchester).** `{t.manchester_template}`\n\n"
            f"**Slots.** {slots or '—'}\n\n"
            f"**Category.** {N.wl(f'ontology/category/{fam}', fam)}"
        )
        rt = era_tables.get(t.template_id)
        if rt and is_cited:
            body += f"  ·  **Relational projection.** {N.wl(f'relational/table/{rt}', rt)}"
        body += "\n"
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
        out.append(N.Note(
            id=term_id, title=t.template_id, kind="ontology-term", data_product="ontology",
            root="current" if is_cited else "archive", body=body, frontmatter={
                "era": "release", "cited": is_cited, "category": fam,
                "qualified_name": f"{t.template_id}@{LEXICON}",
                "slot_types": dict(t.slot_types or {}), "n_chapters": len(chs)}))
    for fam, term_ids in sorted(fams.items()):
        nc = fam_cited.get(fam, 0)
        out.append(N.Note(
            id=f"ontology/category/{fam}", title=fam, kind="ontology-category",
            data_product="ontology", root="current",
            frontmatter={"era": "release", "n_terms": len(term_ids), "n_cited": nc},
            body=(f"**Release-era family** `{fam}` — {len(term_ids)} terms ({nc} cited by the "
                  f"released corpus; the uncited remainder are archive tombstones). Part of the "
                  f"release's {N.wl('lens/terms', 'Lexicon')}.\n\n"
                  + "\n".join(f"- {N.wl(tid, tid.split('/')[-1])}" for tid in sorted(term_ids)))))
    return out


def project_released_ddl(rd: dict) -> list[N.Note]:
    """The RELEASED DDL spine (corpora/ddl, the SHARE record) as current-root relational
    notes — the release kasten's schema surface, one note per released table, grouped by
    the release-era family. Columns/FKs rendered from the run's ddl_statements rows."""
    import json as _json
    out: list[N.Note] = []
    fam_tables: dict[str, list[str]] = {}
    seen: set[str] = set()
    for r in rd["rows"]:
        name = r.get("table_name")
        if not name or name in seen:
            continue
        seen.add(name)
        fam = r.get("family") or "released"
        rid = f"relational/table/{name}"
        fam_tables.setdefault(fam, []).append(rid)
        cols = _json.loads(r.get("columns_json") or "[]")
        col_names = [(c.get("name") if isinstance(c, dict) else c) for c in cols]
        fks = _json.loads(r.get("fks_json") or "[]")
        fk_md = ""
        if fks:
            fk_md = ("\n**Foreign keys.**\n" + "\n".join(
                f"- `{e.get('src_col', e.get('col', '?'))}` → "
                f"`{e.get('dst_table', e.get('ref_table', '?'))}.{e.get('dst_col', e.get('ref_col', 'id'))}`"
                for e in fks) + "\n")
        out.append(N.Note(
            id=rid, title=name, kind="relational-table", data_product="relational",
            frontmatter={"era": "release", "run": rd["run"], "category": fam,
                         "realizes": r.get("template_id"), "n_columns": len(col_names)},
            body=(f"**Released table `{name}`** (run `{rd['run']}`, corpora/ddl — the SHARE "
                  f"record). Realizes {N.wl('ontology/term/' + str(r.get('template_id')), str(r.get('template_id')))}"
                  f" · category {N.wl(f'relational/category/{fam}', fam)}.\n\n"
                  f"Columns ({len(col_names)}): " + " · ".join(f"`{c}`" for c in col_names) + "\n"
                  + fk_md)))
    for fam, rids in sorted(fam_tables.items()):
        out.append(N.Note(
            id=f"relational/category/{fam}", title=f"{fam} (tables)", kind="relational-category",
            data_product="relational", frontmatter={"era": "release", "n_tables": len(rids)},
            body=(f"**Release-era relational category** `{fam}` — {len(rids)} released tables "
                  f"(corpora/ddl `{rd['run']}`).\n\n"
                  + "\n".join(f"- {N.wl(rid, rid.split('/')[-1])}" for rid in sorted(rids)))))
    return out


# ── relational Data Product ─────────────────────────────────────────────────
def project_relational(rows: list[tuple[str, CatalogTemplate]],
                       has_sdg: bool = False) -> list[N.Note]:
    """Project the deterministic DDL spine as relational/table notes.

    Each table carries its typed columns, RI-true sample rows, and its **realized
    subgraph** — the satellites + intra-subgraph FK edges the realize profile expands
    it into (read from the newest spine run). The family-complex cross-family wiring is
    RETIRED: cross-entity edges are the constructs web's to EARN (Convert 2) — the
    earned graph is the verbatim ``relational/sdg-schema`` web, not this spine."""
    import math
    from collections import Counter

    from aegir.ontology.ddl import template_to_table

    spine = []
    for cat, t in rows:
        try:
            spine.append(template_to_table(t, cat))
        except Exception as e:                       # noqa: BLE001
            print(f"  [relational] skip {t.template_id}: {type(e).__name__}: {e}")
    srun = S.spine_run()
    subgraphs = (srun or {}).get("by_template", {})

    # ── Comp 5: confirmation surface — materialize RI-true rows (deterministic, same Comp-4 machinery
    # as the DDL spine: enums + curated pools + LLM-seeded entity values) so the lineup SHOWS sample data
    # and the value/de-canning quality is legible at /lineup, not just asserted by the gate. ───────────
    try:
        from aegir.ontology.chapter_tables import definitions_for_spine, entity_pools_for_spine
        from aegir.ontology.rows import materialize_rows, table_rows_as_records
        materialize_rows(spine, [], definitions=definitions_for_spine(spine),
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

    out: list[N.Note] = []
    cat_tables: dict[str, list[str]] = {}
    for st in spine:
        t, cols = st.template, st.table.columns
        cat = S.relational_category(t)
        rid = f"relational/table/{_table_id(t.template_id)}"
        cat_tables.setdefault(cat, []).append(rid)
        rowsmd = "\n".join(f"| `{c.name}` | {c.slot_type} | {c.slot_ref} |" for c in cols)
        sub = subgraphs.get(t.template_id) or {}
        sats = {n: c for n, c in (sub.get("tables") or {}).items() if n != st.table.name}
        sfks = sub.get("fks") or []
        fk_md = ""
        if sats or sfks:
            fk_md = (f"\n**Realized subgraph** (`{cat}` profile, spine run `{(srun or {}).get('run')}`)"
                     f" — {1 + len(sats)} tables · {len(sfks)} intra-subgraph FK edges:\n\n"
                     + "\n".join(f"- `{n}` ({len(c)} cols)" for n, c in sorted(sats.items()))
                     + ("\n" if sats else "")
                     + ("\n".join(f"- `{e['src_table']}.{e['src_col']}` → `{e['dst_table']}.{e['dst_col']}`"
                                  for e in sfks) + "\n" if sfks else ""))
        else:
            fk_md = "\n_Single-table realize profile (no satellites in the current spine run)._\n"
        if has_sdg:
            fk_md += ("\n_Cross-entity FK edges are the generated web's to earn (Convert 2) — see "
                      + N.wl("relational/sdg-schema", "the SDG schema") + "._\n")

        # ── Comp 5: verbalization + sample rows + quality badge (the confirmation surface) ──
        anchor = (list(t.bfo_anchor_path) or ["(none)"])[-1]
        frames = t.frames()
        vb_md = ""
        if frames:
            extra = f"  _(+{len(frames) - 1} more frames, sampled per chapter)_" if len(frames) > 1 else ""
            vb_md = f"\n**Verbalization.** _{frames[0]}_{extra}\n"

        pk_names = {c.name for c in cols if c.slot_ref == "__pk__"}
        fk_names = {e["src_col"] for e in sfks if e.get("src_table") == st.table.name}
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
            body=body, frontmatter={
                "category": cat, "realizes": t.template_id, "table_name": st.table.name,
                "columns": [{"name": c.name, "type": c.slot_type, "slot": c.slot_ref} for c in cols],
                "not_null": sorted(getattr(st, "not_null", set()) or set()),
                "quality": quality,
                "realized_subgraph": ({"run": (srun or {}).get("run"),
                                       "n_tables": 1 + len(sats), "n_fks": len(sfks),
                                       "satellites": sorted(sats)} if (sats or sfks) else {})}))

    _CAT_GLOSS = {
        "junction": "many-to-many participation lowered to a junction table",
        "dimension": "a star profile — the entity as a dimension with satellites",
        "nested-child": "a parent-owned child table (composition)",
        "constraint": "a CHECK/cardinality-bearing normalized table",
        "enum": "a closed value-set lowered to an enum/lookup",
        "eav": "an open attribute set lowered entity-attribute-value",
        "intermediate": "intermediate domain classes (CTA subsumers) — plain normalized tables",
    }
    for cat, rids in sorted(cat_tables.items()):
        gloss = _CAT_GLOSS.get(cat, "")
        body = (f"Relational tables grounding the **{cat}** DDL shape — {len(rids)}."
                + (f" _{gloss}._" if gloss else "") + "\n\n"
                + "\n".join(f"- {N.wl(rid, rid.split('/')[-1])}" for rid in sorted(rids)))
        out.append(N.Note(id=f"relational/category/{cat}", title=f"{cat} (tables)",
                          kind="relational-category", data_product="relational", body=body,
                          frontmatter={"category": cat, "n_tables": len(rids)}))
    n_sfks = sum(len(s.get("fks") or []) for s in subgraphs.values())
    print(f"  relational: {len(spine)} tables in {len(cat_tables)} grounds-shape categories; "
          f"realized subgraphs {'from ' + srun['run'] + f' ({n_sfks} intra-subgraph FKs)' if srun else 'absent (no spine run)'}")
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
            kind="content-chapter", data_product="content", root="scratch", body=body0 + "\n", frontmatter={
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
                kind="content-chapter", data_product="content", root="scratch", body=f"{nav}\n\n{wins[k]}\n",
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
                   topic_colls: dict[int, list[str]] | None = None,
                   tid2cat: dict[str, str] | None = None,
                   retired_cat: dict[str, str] | None = None) -> list[N.Note]:
    topic_chapters = topic_chapters or {}
    topic_colls = topic_colls or {}
    tid2cat = tid2cat or {}
    retired_cat = retired_cat or {}
    out: list[N.Note] = []
    by_status: dict[str, list[int]] = {}
    for r in recs:
        tid = int(r["topic_id"])
        by_status.setdefault(str(r.get("status")), []).append(tid)
        # The category resolves through the top template's LIVE category (the coverage rows'
        # stored top_family is the retired file-stem axis). A coverage run computed against
        # the retired catalog resolves to the archive tombstone instead — a real trail, not
        # a dangling link (the cure remains re-running the coverage audit).
        top = r.get("top_template_id")
        cat = tid2cat.get(str(top)) if top else None
        rfam = retired_cat.get(str(top)) if (top and not cat) else None
        edges = []
        if top and (cat or rfam):
            edges.append(f"**Nearest term.** {N.wl(f'ontology/term/{top}', str(top))}")
        elif top:
            edges.append(f"**Nearest term** (unknown catalog): `{top}`")
        if cat or rfam:
            c = cat or rfam
            edges.append(f"**Top category.** {N.wl(f'ontology/category/{c}', c)}")
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
                "top_category": cat or rfam, "top_term": top, "era_top": bool(rfam),
                "n_chapters": len(covered)}))
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


def assign_topic_threads(corpus: list[dict], coverage: list[dict]) -> "dict[str, list[int]]":
    """MUTATES corpus records in place: template-driven chapters get an ORGANIC
    ``target_topic_id`` (+ style thread) voted from their templates' coverage alignment
    — without this the topic/collections axis collapses (most records carry no topic).
    Returns ``term_topics`` (template → aligned topic ids). SHARED by ``run()`` and
    ``aegir.viz.lineup_data`` — the two substrates must stay in lock-step (this function
    exists because they silently didn't: the chord app saw 1 collection for weeks)."""
    term_topics: dict[str, list[int]] = {}
    for r in coverage:
        if int(r["topic_id"]) < 0:            # BERTopic outlier/noise cluster (-1) — never assign
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
    for r in corpus:
        cur = r.get("target_topic_id")
        if cur is not None and int(cur) < 0:
            r["target_topic_id"] = cur = None
        if r.get("style_topic_ids"):
            r["style_topic_ids"] = [s for s in _idlist(r["style_topic_ids"]) if int(s) >= 0]
        if cur is None:
            votes: dict[int, int] = {}
            for tid in (r.get("template_ids") or []):
                for tp in term_topics.get(str(tid), []):
                    votes[tp] = votes.get(tp, 0) + 1
            if votes:
                ranked = sorted(votes, key=lambda k: (-votes[k], k))
                r["target_topic_id"] = ranked[0]
                r["style_topic_ids"] = ranked[1:5]
    return term_topics


def project_collections(recs: list[dict], coverage: list[dict],
                         live_terms: set | None = None,
                         cgraph: "dict | None" = None) -> tuple[list[N.Note], dict]:
    """De-flattened, many-to-many collections — the unit the landing pivots. A collection's
    DOCUMENTS (chapters) are the hub: it relates to MANY topics (target ∪ style across its
    chapters) and many terms/tables, and each terminal recurs across collections. Returns
    (notes, maps); maps drive the lens pivots and the transpose backlinks. ``live_terms``
    gates the Underlying-tables links (retired terms resolve as archive tombstones but have
    no spine table)."""
    by_coll: dict[int, list[tuple[int, dict]]] = {}
    for i, r in enumerate(recs):
        t = r.get("target_topic_id")
        if t is not None:
            by_coll.setdefault(int(t), []).append((i, r))
    cov = {int(c["topic_id"]): c for c in coverage}
    coll_terms: dict[str, list[str]] = {}
    coll_topics: dict[str, list[int]] = {}
    coll_chapters: dict[str, list[str]] = {}
    coll_tables: dict[str, list[str]] = {}
    notes: list[N.Note] = []
    for tid in sorted(by_coll):
        chs = by_coll[tid]
        cid = f"collection/topic-{tid:03d}"
        cids = [_chapter_id(r, i) for i, r in chs]
        topics = sorted({tid} | {int(s) for _, r in chs for s in _idlist(r.get("style_topic_ids"))})
        terms = sorted({str(x) for _, r in chs for x in _idlist(r.get("template_ids"))})
        coll_terms[cid], coll_topics[cid], coll_chapters[cid] = terms, topics, cids
        ctables: "list[str]" = []
        if cgraph:
            h6s = {m.group(1) for c in cids if (m := re.search(r"_([0-9a-f]{6})\.", c))}
            ctables = sorted({t for h in h6s for t in cgraph.get("h6_tables", {}).get(h, ())})
        coll_tables[cid] = ctables
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
            + "**Underlying tables.** " + ((" · ".join(
                N.wl(f"relational/table/{t}", t) for t in ctables[:24]))
                if ctables else ((" · ".join(
                N.wl(((cgraph or {}).get("archived_terms", {}).get(x) or {}).get("id", f"relational/table/{_table_id(x)}"), x)
                for x in terms[:24]) + " _(archived kasten)_")
                if (cgraph or {}).get("archived_terms") else "—"))
            + "\n\n" + f"**Cross-reference.** {N.wl(cid + '/terms', 'tables × data-elements')} "
            + "(each table's columns as ontology-terms).\n")
        notes.append(N.Note(id=cid, title=f"collection · topic {tid}", kind="collection",
                            data_product="content", root="scratch", body=body, frontmatter={
                                "topic_id": tid, "n_topics": len(topics), "n_terms": len(terms),
                                "n_chapters": len(cids), "topics": topics}))
    if cgraph:
        tcols = cgraph.get("table_cols", {})
        telems = cgraph.get("term_elements", {})
        for cid in coll_tables:
            ctabs, terms_ = coll_tables[cid], coll_terms.get(cid, [])
            tid = int(cid.rsplit("-", 1)[1])
            rows = ["| table | data-elements (ontology-terms) |", "|---|---|"]
            if ctabs:                                   # construct-era: live spine tables
                for t in ctabs[:30]:
                    cons = sorted({c for _, c in tcols.get(t, ()) if c})
                    cell = " · ".join(N.wl(f"ontology/term/{c}", c) for c in cons[:10]) or "—"
                    rows.append(f"| {N.wl(f'relational/table/{t}', t)} | {cell} |")
                caveat = ""
            else:                                       # refined-era: tables were template-derived
                arch = cgraph.get("archived_terms", {}) if cgraph else {}
                era = next(iter(arch.values()))["id"].split("/ontology/")[0] if arch else None
                for x in terms_[:30]:
                    els = telems.get(x) or (arch.get(x) or {}).get("slots") or []
                    cell = " · ".join(f"`{e}`" for e in els[:10]) or "—"
                    tcell = (N.wl(arch[x]["id"], x) + f" _({era} kasten)_" if x in arch else f"`{x}`")
                    rows.append(f"| {tcell} · {N.wl(f'ontology/term/{x}', 'term')} | {cell} |")
                caveat = ((f"\n_These chapters embed template-derived tables from the "
                           f"{era} era — table links open that kasten's archived term panels; "
                           "data-elements are the terms' slot types._\n") if era else "")
            if len(rows) <= 2:
                continue
            notes.append(N.Note(
                id=f"{cid}/terms", title=f"collection · topic {tid} — terms",
                kind="collection-terms", data_product="content", root="scratch",
                frontmatter={"topic_id": tid, "n_tables": len(ctabs) or len(terms_)},
                links=[cid] + [f"relational/table/{t}" for t in ctabs[:30]],
                body=(f"**Tables × data-elements** for {N.wl(cid, f'collection topic {tid}')} — "
                      "each embedded table's columns, embodied as ontology-terms.\n\n"
                      + "\n".join(rows) + "\n" + caveat)))
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
                        data_product="content", root="scratch", body=idx,
                        frontmatter={"n_collections": len(by_coll)}))
    table_colls: dict[str, list[str]] = {}
    for cid, ctabs in coll_tables.items():
        for t in ctabs:
            table_colls.setdefault(t, []).append(cid)
    return notes, {"collections": sorted(by_coll), "coll_terms": coll_terms,
                   "coll_topics": coll_topics, "topic_colls": topic_colls, "term_colls": term_colls,
                   "coll_tables": coll_tables, "table_colls": table_colls}


# ── lenses (the landing pivot: collections × the lens-selected axis) ──────────
def _relational_spine(rows):
    """(tid→table_name, edge list, tid→column-token set) for the Schema chord substrate.

    Post family-complex retirement the deterministic subgraphs are DISJOINT across
    templates (0 shared table names, 0 cross-template FKs — measured), so the chord's
    discriminating schema signal is COLUMN VOCABULARY (typed attributes genuinely recur
    across templates — the h_colset/de-canning axis), with the realize subgraph edges
    kept for the FK-hub term. Guarded + lazy: the chord is optional enrichment, so any
    failure yields ({}, [], {}) rather than breaking the build."""
    from types import SimpleNamespace
    tid_table: dict[str, str] = {}
    fks: list = []
    col_tokens: dict[str, set] = {}
    srun = S.spine_run()
    if srun:
        for tid, sub in srun["by_template"].items():
            tables = sub.get("tables") or {}
            if not tables:
                continue
            primary = next(iter(tables))
            tid_table[tid] = primary
            col_tokens[tid] = {(c.get("name") if isinstance(c, dict) else c)
                               for cols in tables.values() for c in cols} - {"id", None}
            fks += [SimpleNamespace(src_table=primary, dst_table=n)
                    for n in tables if n != primary]
        return tid_table, fks, col_tokens
    # No spine run on disk — fall back to the in-process one-table lowering (columns only).
    from aegir.ontology.ddl import template_to_table
    for cat, t in rows:
        try:
            st = template_to_table(t, cat)
        except Exception:  # noqa: BLE001
            continue
        tid_table[t.template_id] = st.table.name
        col_tokens[t.template_id] = {c.name for c in st.table.columns if c.name != "id"}
    return tid_table, fks, col_tokens


def project_lenses(era_fams: list[str], has_content: bool, has_topics: bool,
                   maps: dict | None = None, rd: dict | None = None,
                   live_terms: set | None = None,
                   latest_release: str | None = None,
                   rel_colls: "list[tuple[str, str, int]] | None" = None,
                   rel_schema: "dict[str, list[str]] | None" = None) -> list[N.Note]:
    """The RELEASE kasten's lenses (current root) — the lens surfaces over the released
    corpus, its era lexicon, and the released DDL spine. The trunk (scratch) gets its own
    lens set from :func:`project_trunk_lenses` — same ids, root-resolved (roots are refs)."""
    maps = maps or {}
    colls = maps.get("collections")
    live_terms = live_terms or set()
    # terms (default): collections × realized terms — the grounding pivot. The RELEASE's
    # anchor-concept collections (RH: topics ≡ concept anchors) front the kasten when present.
    if rel_colls:
        rows = ["| collection (anchor concept) | chapters |", "|---|---|"]
        for cid_, label_, n_ in rel_colls[:24]:
            rows.append(f"| {N.wl(cid_, label_[:64])} | {n_} |")
        terms_body = ("**Lexicon × Collections (release).** The v-current release's anchor-concept "
                      "collections — one passage, one topic; each collection cross-references its "
                      "live spine tables and their column concepts.\n\n" + "\n".join(rows)
                      + f"\n\nAll: {N.wl('collection/index', 'collections')}")
        terms = N.Note(id="lens/terms", title="Lexicon × Collections", kind="lens",
                       data_product="ontology", frontmatter={"lens": "terms", "lexicon": LEXICON},
                       body=terms_body)
    elif colls:
        rows = ["| collection | realizes terms |", "|---|---|"]
        for tid in colls:
            cid = f"collection/topic-{tid:03d}"
            tms = sorted(maps["coll_terms"][cid], key=lambda x: (x not in live_terms, x))
            cell = " · ".join(N.wl(f"ontology/term/{x}", x) for x in tms[:6]) + (f" …(+{len(tms) - 6})" if len(tms) > 6 else "")
            rows.append(f"| {N.wl(cid, f'topic {tid}')} | {cell or '—'} |")
        terms_body = ("**Lexicon × Collections.** Each collection (a topic-grounded bundle) and the "
                      "ontology terms it realizes — the grounding made visible. Click a term to pivot "
                      "to *its* collections; a collection for its full bundle.\n\n" + "\n".join(rows))
    else:
        terms_body = ("**Lexicon** `" + LEXICON + "` (release era). Browse by family:\n\n"
                      + "\n".join(f"- {N.wl(f'ontology/category/{c}', c)}" for c in era_fams))
    if not rel_colls:
        terms = N.Note(id="lens/terms", title="Lexicon × Collections" if colls else "Lexicon",
                       kind="lens", data_product="ontology",
                       frontmatter={"lens": "terms", "lexicon": LEXICON}, body=terms_body)
    # schema: the RELEASE's relational surface — collections × their live tables (the promised
    # pivot), the spine record, and the most-shared tables as entry points (RH 2026-07-20: the
    # lens must carry ACTUAL schema, not a pointer at a pointer).
    schema_body = ("**Schema × Collections.** The released relational footprint — a base table is "
                   "shared across the collections whose chapters embed views over it.\n")
    rel_schema = rel_schema or {}
    if rel_schema:
        rows = ["| collection | live tables |", "|---|---|"]
        for cid_, label_, _n in (rel_colls or [])[:20]:
            tabs_ = rel_schema.get(cid_, [])
            cell = " · ".join(N.wl(f"relational/table/{t}", t) for t in tabs_[:8]) \
                   + (f" _…+{len(tabs_) - 8}_" if len(tabs_) > 8 else "")
            rows.append(f"| {N.wl(cid_, label_[:48])} | {cell or '—'} |")
        schema_body += "\n" + "\n".join(rows) + "\n"
        shared: dict = {}
        for cid_, tabs_ in rel_schema.items():
            for t in tabs_:
                shared[t] = shared.get(t, 0) + 1
        hubs = sorted(shared.items(), key=lambda kv: -kv[1])[:12]
        if hubs:
            schema_body += ("\n**Most-shared tables** (collection count): "
                            + " · ".join(f"{N.wl(f'relational/table/{t}', t)} ({n_})"
                                         for t, n_ in hubs) + "\n")
    if rd:
        schema_body += (f"\n**The released DDL spine**: corpora/ddl `{rd['run']}` — "
                        f"{len(rd['rows'])} tables, gate-certified "
                        f"({N.wl('release/v' + (latest_release or ''), 'release record') if latest_release else 'release record'}).\n")
    else:
        schema_body += "\n_No released DDL spine on disk (corpora submodule absent)._\n"
    schema_body += "\n_Trunk schema work (the generated web + the live spine) lives in `scratch`._"
    schema = N.Note(id="lens/schema", title="Schema × Collections" if colls else "Schema", kind="lens",
                    data_product="relational", frontmatter={"lens": "schema", "lexicon": LEXICON},
                    body=schema_body)
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
                        + f"\n\nAll: {N.wl('collection/index', 'collections')} · {N.wl('topic/index', 'topics')}"
                        + f" · {N.wl('content/index', 'chapters')}")
    else:
        content_body = ("**Content.** The corpus + FinePDFs topics.\n\n"
                        + (f"- {N.wl('content/index', 'the corpus chapters')}\n- {N.wl('topic/index', 'the FinePDFs topics')}"
                           if has_content or has_topics else "No corpus/topics projected yet."))
    if rel_colls:
        content_body = ("**Content (release).** The release's chapters, grouped by anchor-concept "
                        "collection:\n\n"
                        + "\n".join(f"- {N.wl(cid_, label_[:64])} ({n_} ch)"
                                     for cid_, label_, n_ in rel_colls[:20])
                        + f"\n\nAll: {N.wl('collection/index', 'collections')} · "
                        + N.wl("content/release-chapters", "release chapters"))
    if latest_release:
        content_body += ("\n\n**Release record.** "
                         + N.wl(f"release/v{latest_release}", f"v{latest_release}")
                         + " (the sdg-corpora card this kasten projects)")
    content = N.Note(id="lens/content", title="Content × Topics" if colls else "Content", kind="lens",
                     data_product="content", frontmatter={"lens": "content"}, body=content_body)

    # The lens chords render live via the bokeh server (aegir.viz.lineup_app), embedded by the React
    # <PanelView> — no chord is baked into the note frontmatter anymore.
    return [terms, schema, content]




def project_lexicon_constructs() -> list[N.Note]:
    """The Lexicon's META-VOCABULARY as definitional notes (RH 2026-07-19): each construct carries its
    CURRENT operational definition + the machinery that enforces it + its relations — definitions that
    compile, not essays. Root=scratch (trunk): these are live definitions under refinement; the lens
    guard is that refinement lands HERE (and in the modules), keeping the lineup a Cunningham primitive,
    not a wiki."""
    C = {
        "domain": ("A DOMAIN is a SPECIFICATION over concepts — a region of admissible input meaning "
                   "that naturally SUBSUMES many SKOS concepts, while any concept may occur within many "
                   "domains: the relationship is a proper MANY-TO-MANY LATTICE, not a tree (RH "
                   "2026-07-19 — domains are NOT SKOS top-concepts, and navigation must not imply "
                   "single-parenting).\n\nOperationalization: a domain is realized through the "
                   + N.wl("lexicon/construct/aperture", "Canonical Aperture") + "'s composite anchors "
                   "and the admission rule (`aegir.ontology.domain_index`). Domains may be refined "
                   "IN SITU — e.g. as novel physical compute environments are explored — and any "
                   "refinement must preserve the sufficiency-of-differentiation of the aperture's "
                   "constituent concepts."),
        "concept": ("A CONCEPT is one anchor point in the vocab collection — an ontology-derived SKOS "
                    "concept embedded in qdrant: the unit of classification and the identity-bearer "
                    "for anchor-topics. A concept naturally occurs within MULTIPLE domains "
                    "(the domain⇄concept lattice is many-to-many).\n\nOperationalization: "
                    "`domain_index` vocab collection; strategy `lens/vocab.snapshot`. Relations: a "
                    "topic IS a concept once passages accumulate on it; PRIMARY constituent concepts "
                    "composite into the Canonical Aperture's anchors; spectral analysis may PROPOSE "
                    "new concepts (the aperture-refinement loop)."),
        "item": ("An ITEM is the SOURCE-AGNOSTIC resolution unit of the lexicon layer: a segment "
                 "sized to the qdrant-maxsim token limits, produced by ONE itemization procedure "
                 "regardless of origin. A DOCUMENT is a SEQUENCE of items (the structural concept "
                 "borrowed from FedWiki) — including embedded views and transitional passages. INPUT "
                 "items resolve from open-domain corpora (e.g. FinePDFs); OUTPUT items resolve from "
                 "generated corpora (e.g. chapters) against the Canonical Aperture via the SAME "
                 "procedures employed during input admission (RH 2026-07-19).\n\nOutput-side "
                 "resolution is CLASSIFICATORY, not exclusionary: a SLIDING WINDOW ensures "
                 "anchor-concept-associated (DEFINITE) items are not missed, and any document with a "
                 "definite item almost necessarily has INDEFINITE-item siblings — views, transitions — "
                 "which REMAIN VALID OUTPUT CONTENT (indefinite ≠ drift; RH relaxation 2026-07-19).\n\n"
                 "Consequence: input-side and output-side analysis (anchor-topics, spectral topics, "
                 "congruence — measured over definite items) is commensurable BY CONSTRUCTION — "
                 "same resolution, same aperture, same encoder."),
        "topic": ("A TOPIC is a concept anchor with accumulated evidence: each passage assigns to its "
                  "nearest concept anchor, margin-gated — ONE passage, ONE topic (ratified 2026-07-07; "
                  "collection-state pinned; BERTopic vestigial).\n\n_De novo_ topics (STAGED, #30): "
                  "CLT/SAE spectral communities over feature co-activation across ITEMS — input items "
                  "(open-domain corpora, e.g. FinePDFs) and output items (generated corpora, e.g. "
                  "chapters), both aperture-resolved by the same procedures: commensurable BY "
                  "CONSTRUCTION. Reconciliation with anchor-topics may propose new "
                  "concepts → aperture refinement. Operationalization: the inverted topic layer; "
                  "instruments: clt-qwen3-1.7b (live) + SAE-Res-Qwen3.5-27B."),
        "aperture": ("The CANONICAL APERTURE is the composite admission surface: a qdrant collection "
                     "whose anchors COMPOSITE primary constituent concepts (currently named 'aiming' — "
                     "an odd historical name; migration to aperture-proper naming is deliberate "
                     "follow-up #31, not a drive-by). A passage is admitted iff its ColBERT/MaxSim "
                     "score against the composite anchors clears the threshold.\n\nThe VERIFICATION "
                     "obligation (RH 2026-07-19): the primary constituent concepts must be SUFFICIENTLY "
                     "DIFFERENTIATED under ANY domain specification or refinement — the "
                     "insufficient-differentia frame extended from taxonomy species to aperture "
                     "constituents, measurable with the same margin machinery (genus_induction/"
                     "discriminability). Refinement may happen in situ (novel physical compute "
                     "environments). Constituent-concept mappings are not yet surfaced in the strategy "
                     "snapshot (vector_sha only) — surfacing them is part of #31."),
        "collection": ("A COLLECTION is an output-side bundle: the documents whose tables/views form "
                       "one connected DDL graph (FK + view-composition edges) after infrastructure-hub "
                       "removal — the Barabási-calibrated operating point.\n\nOperationalization: "
                       "`targets/collections_unit` (strategy) + `scripts/relational_collections.py`. "
                       "Relations: realizes terms; threaded by topics; the chapter's relational home."),
        "term": ("A TERM is a catalog lexeme: a template head with its Manchester axiom skeleton, "
                 "verbalized via the frame set; in Atlas-glossary organization a Term under a Category "
                 "under the Lexicon.\n\nOperationalization: `catalog.json` + verbalization frames. "
                 "Relations: grounded to a BFO/CCO anchor; realized by collections; the aggregate "
                 "lexicon's DISCRIMINATING POTENTIAL calibrates taxonomy depth (the NHSVM "
                 "rate-distortion constraint)."),
        "genus": ("A GENUS is an induced mid-tier class at the rate-distortion FRONTIER (frontier_k "
                  "from the lexicon's discriminating potential): the precondition for differentia — "
                  "a genus alone is a bag; a taxonomy is genus + differentiated species.\n\n"
                  "Operationalization: `aegir.ontology.genus_induction` (frontier detection) + the "
                  "authoring CAS loop. Relations: species SubClassOf genus; disjointness within."),
        "differentia": ("A DIFFERENTIA is the OBSERVABLE discriminating restriction a species earns "
                        "under its genus (propose → membranes → HermiT), projecting to a data column "
                        "or FK join: authentic-taxonomy ≡ sufficient-differentia ≡ elucidable.\n\n"
                        "Operationalization: `aegir.ontology.differentia_authoring`; the SHACL shapes "
                        "carry it (sh:in domains render as table Constraints in the lineup)."),
    }
    sibs = list(C)
    notes = [N.Note(
        id="lexicon/constructs", title="Lexicon constructs", kind="lexicon-construct",
        data_product="ontology", root="scratch",
        body=("**The meta-vocabulary of the pipeline** — refined here, enforced in the modules.\n\n"
              "The chain: the CANONICAL APERTURE (composite anchors over primary constituent "
              "concepts) admits INPUT ITEMS (open-domain corpora) → TOPICS accumulate on concept "
              "anchors → CHAPTERS generate against terms and RESOLVE BACK to OUTPUT ITEMS through "
              "the same aperture → COLLECTIONS bundle the relational output. The ontology stratum "
              "(TERMS · GENERA · DIFFERENTIAE) threads every step; the refinement loop closes when "
              "spectral topics propose new concepts.\n\nSTRUCTURAL COMMITMENT (RH): domain⇄concept "
              "is a MANY-TO-MANY LATTICE — a domain subsumes many concepts, a concept occurs in many "
              "domains — so Lexicon navigation is FACETED, never single-parent tree.\n\n"
              + "\n".join(f"- {N.wl(f'lexicon/construct/{c}', c)}" for c in sibs)))]
    for c, body in C.items():
        notes.append(N.Note(
            id=f"lexicon/construct/{c}", title=c, kind="lexicon-construct",
            data_product="ontology", root="scratch",
            links=[f"lexicon/construct/{x}" for x in sibs if x != c],
            body=body + "\n\nSiblings: "
                 + " · ".join(N.wl(f"lexicon/construct/{x}", x) for x in sibs if x != c)))
    return notes

def _aperture_constituents_md(p0: dict, label_to_aid: "dict[str, str] | None" = None) -> str:
    """The anchor's lattice, from the snapshot payload: primary concept constituents (α-banded,
    rel-scored) + adjacent domains typed apart. Honest placeholder when a pre-#31 snapshot lacks it."""
    cons = p0.get("constituents") or []
    concepts = [x for x in cons if x.get("kind") == "concept"]
    adj = [x for x in cons if x.get("kind") == "adjacent-domain"]
    if not cons:
        return ("Composites primary constituent concepts; the constituent list is not in this "
                "strategy snapshot (re-seed after build_aperture_constituents).\n\n")
    label_to_aid = label_to_aid or {}
    out = f"**Primary constituent concepts** ({len(concepts)}, α-banded, rel to best):\n\n"
    out += "\n".join(
        f"- {N.wl('lexicon/concept/' + str(x.get('code')), (x.get('label') or '')[:52])} · rel {x.get('rel')}"
        if x.get("code") else f"- {x.get('label')} · rel {x.get('rel')}"
        for x in concepts[:16])
    if len(concepts) > 16:
        out += f"\n- _…{len(concepts) - 16} more_"
    if adj:
        out += ("\n\n**Adjacent domains** (retrieval-adjacent siblings — adjacency, not "
                "constituency):\n\n"
                + "\n".join(
                    (f"- {N.wl('lexicon/aperture/' + label_to_aid[x.get('label')], x.get('label'))} "
                     f"· rel {x.get('rel')}") if x.get("label") in label_to_aid
                    else f"- {x.get('label')} · rel {x.get('rel')}"
                    for x in adj[:8]))
    return out + "\n\n"





def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:48]


def project_release_collections(sc: "dict | None", rel_by_pid: "dict[str, list[str]]",
                                live_ids: "set[str] | None" = None
                                ) -> "tuple[list[N.Note], dict[str, list[str]]]":
    """The RELEASE's collections (current root), keyed by ANCHOR CONCEPTS — the inverted
    topic layer made structural (RH: topics ≡ qdrant concept anchors; one passage → one
    topic; the BERTopic topic-NNN bundles are trunk-era vestige and live in scratch now).
    concept_graph.json carries passage→concept MaxSim edges; each chapter joins the
    collection of its passage's DOMINANT concept. Tables come from the chapters' own
    constructs — the live spine, so every cross-reference here resolves without era hops.
    Returns (notes, table→collection-ids) for the table panels' Collections sections."""
    root = S.sdg_run_root()
    if not (sc and rel_by_pid and root and (root / "concept_graph.json").exists()):
        return [], {}
    try:
        cg = json.loads((root / "concept_graph.json").read_text())
    except Exception:  # noqa: BLE001
        return [], {}
    labels = {n["id"]: n.get("label") or n["id"] for n in cg.get("nodes", [])
              if n.get("kind") == "concept"}
    top: "dict[str, tuple[str, float]]" = {}
    for e in cg.get("edges", []):
        src, dst = e.get("src", ""), e.get("dst", "")
        if not (src.startswith("passage:") and dst.startswith("concept:")):
            continue
        pid = src.split(":", 1)[1]
        sc_ = float(e.get("score") or 0)
        if pid not in top or sc_ > top[pid][1]:
            top[pid] = (dst, sc_)
    pid_tables: "dict[str, list[str]]" = {}
    table_concepts: "dict[str, list[str]]" = {}
    for tn, te in (sc.get("tables") or {}).items():
        table_concepts[tn] = sorted({c.get("concept") for c in te.get("columns", [])
                                     if c.get("concept")})
    for tn, te in (sc.get("tables") or {}).items():
        for pid in te.get("constructs", []):
            pid_tables.setdefault(pid, []).append(tn)
    by_concept: "dict[str, list[str]]" = {}
    for pid in rel_by_pid:
        cnode = top.get(pid, (None, 0))[0]
        if cnode:
            by_concept.setdefault(cnode, []).append(pid)
    notes: "list[N.Note]" = []
    table_colls: "dict[str, list[str]]" = {}
    coll_ids = []
    for cnode, pids in sorted(by_concept.items(), key=lambda kv: -len(kv[1])):
        label = labels.get(cnode, cnode)
        cid = f"collection/anchor-{_slug(label)}"
        coll_ids.append((cid, label, pids))
        tabs = sorted({t for p in pids for t in pid_tables.get(p, ())})
        for t in tabs:
            table_colls.setdefault(t, []).append(cid)
        chs = [c for p in sorted(pids) for c in rel_by_pid.get(p, [])]
        live_ids = live_ids or set()
        def _term(x: str) -> str:
            sn = re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()
            return N.wl(f"ontology/term/{sn}", x) if sn in live_ids else f"`{x}`"
        terms = sorted({c for t in tabs for c in table_concepts.get(t, ())} | set(tabs))
        body = (
            f"**Anchor-concept collection** — {N.wl('lexicon/construct/topic', 'anchor-topic')} "
            f"“{label}” ({cnode}); {len(pids)} passages · {len(chs)} chapters · "
            f"{len(tabs)} live tables.\n\n"
            "**Documents.** " + (" · ".join(
                N.wl(c, c.split("_")[-1][:12].replace(".", " · ")) for c in chs[:16]) or "—")
            + (f" _…+{len(chs) - 16}_" if len(chs) > 16 else "") + "\n\n"
            "**Underlying tables.** " + (" · ".join(
                N.wl(f"relational/table/{t}", t) for t in tabs[:18]) or "—") + "\n\n"
            "**Realizes terms.** " + (" · ".join(_term(x) for x in terms[:18]) or "—")
            + (f" _…+{len(terms) - 18}_" if len(terms) > 18 else "") + "\n\n"
            f"**Cross-reference.** {N.wl(cid + '/terms', 'tables × data-elements')}\n")
        notes.append(N.Note(id=cid, title=f"collection · {label[:60]}", kind="collection",
                            data_product="content", root="current",
                            links=[f"relational/table/{t}" for t in tabs[:18]] + chs[:16],
                            body=body, frontmatter={"anchor_concept": cnode, "label": label,
                                                    "n_chapters": len(chs), "n_tables": len(tabs)}))
        rows = ["| table | data-elements (ontology-terms) |", "|---|---|"]
        for t in tabs[:30]:
            cell = " · ".join(_term(c) for c in table_concepts.get(t, [])[:10]) or "—"
            rows.append(f"| {N.wl(f'relational/table/{t}', t)} | {cell} |")
        if len(rows) > 2:
            notes.append(N.Note(
                id=f"{cid}/terms", title=f"{label[:48]} — terms", kind="collection-terms",
                data_product="content", root="current",
                links=[cid] + [f"relational/table/{t}" for t in tabs[:30]],
                body=(f"**Tables × data-elements** for {N.wl(cid, label[:60])} — live spine "
                      "tables, columns embodied as ontology-terms.\n\n" + "\n".join(rows) + "\n")))
    if coll_ids:
        idx = (f"**Collections** — {len(coll_ids)} anchor-concept bundles over the release "
               "(topics ≡ concept anchors; one passage → one topic).\n\n"
               + "\n".join(f"- {N.wl(cid, label[:64])} ({len(pids)} passages)"
                            for cid, label, pids in coll_ids))
        notes.append(N.Note(id="collection/index", title="Collections", kind="collection-index",
                            data_product="content", root="current", body=idx,
                            frontmatter={"n_collections": len(coll_ids)}))
    return notes, table_colls


def project_release_chapters(sc: "dict | None") -> "tuple[list[N.Note], dict]":
    """The RELEASED corpus's chapters as current-root content notes (RH UXR fold-in): current is
    the latest release kasten, so the release's own prose belongs there — and it is the generation
    that actually embeds the live relational spine (construct-hash join), which the refined path-a
    corpus cannot (different generation, retired tables). Returns (notes, pid→[note ids])."""
    root = S.sdg_run_root()
    if root is None or not (root / "chapters").exists():
        return [], {}
    # per-construct link context: the chapter panel sits on the graph's richest join point
    pid_tables: "dict[str, list[str]]" = {}
    for tn, te in (sc.get("tables") or {}).items():
        for pid_ in te.get("constructs", []):
            pid_tables.setdefault(pid_, []).append(tn)
    pid_views: "dict[str, list[str]]" = {}
    for vn, ve in (sc.get("views") or {}).items():
        for pid_ in (ve.get("constructs") or [ve.get("construct")]):
            if pid_:
                pid_views.setdefault(pid_, []).append(vn)

    notes: "list[N.Note]" = []
    by_pid: "dict[str, list[str]]" = {}
    for cdir in sorted((root / "chapters").iterdir()):
        if not cdir.is_dir():
            continue
        pid = cdir.name
        for reg in ("natural", "semantic"):
            f = cdir / f"{reg}.md"
            if not f.exists():
                continue
            try:
                text = f.read_text()
            except Exception:  # noqa: BLE001
                continue
            # title: H1, else the first section heading, else the opening sentence — never a bare hash
            m = (re.search(r"^#\s+(.+)$", text, re.M)
                 or re.search(r"^##\s+(.+)$", text, re.M)
                 or re.search(r"^([A-Z][^.\n]{10,90})[.\n]", text, re.M))
            _tabs0 = sorted(pid_tables.get(pid, []))
            fallback = (_tabs0[0].replace("_", " ") + " (chapter)") if _tabs0 else pid[:12]
            title = (m.group(1).strip()[:72] if m else fallback) + f" · {reg}"
            nid = f"content/chapter/rel_{pid[:12]}.{reg}"
            twin = f"content/chapter/rel_{pid[:12]}.{'natural' if reg == 'semantic' else 'semantic'}"
            tabs = sorted(pid_tables.get(pid, []))
            vws = sorted(pid_views.get(pid, []))
            head = (f"*Release chapter — **{reg}** register · construct `{pid[:12]}` · "
                    f"{N.wl(twin, ('natural' if reg == 'semantic' else 'semantic') + ' register')}*\n\n"
                    + (("**Tables.** " + " · ".join(N.wl(f"relational/table/{t}", t) for t in tabs[:10])
                        + (f" _…+{len(tabs) - 10}_" if len(tabs) > 10 else "") + "\n\n") if tabs else "")
                    + (("**Views.** " + " · ".join(N.wl(f"relational/view/{v}", v) for v in vws[:6])
                        + (f" _…+{len(vws) - 6}_" if len(vws) > 6 else "") + "\n\n") if vws else ""))
            wins = [w + ("\n```" if w.count("```") % 2 else "") for w in _windows(text, 12000)]
            n = len(wins)
            body0 = head + wins[0]
            if n > 1:
                body0 += f"\n\n→ {N.wl(nid + '__w2', f'continue — window 2 of {n}')}"
            notes.append(N.Note(
                id=nid, title=title + (f" (1/{n})" if n > 1 else ""), kind="content-chapter",
                data_product="content", root="current",
                links=[twin] + [f"relational/table/{t}" for t in tabs[:10]]
                      + [f"relational/view/{v}" for v in vws[:6]],
                frontmatter={"register": reg, "construct": pid, "release": True, "windows": n},
                body=body0 + "\n"))
            for k in range(1, n):
                prev = nid if k == 1 else f"{nid}__w{k}"
                nav = f"← {N.wl(prev, f'window {k} of {n}')}"
                if k + 1 < n:
                    nav += f" · → {N.wl(f'{nid}__w{k + 2}', f'window {k + 2} of {n}')}"
                notes.append(N.Note(
                    id=f"{nid}__w{k + 1}", title=f"{title} ({k + 1}/{n})", kind="content-chapter",
                    data_product="content", root="current",
                    frontmatter={"register": reg, "construct": pid, "release": True},
                    body=f"{nav}\n\n{wins[k]}\n"))
            by_pid.setdefault(pid, []).append(nid)
    if notes:
        idx_body = (f"**Release chapters** — {len(notes)} notes over {len(by_pid)} constructs "
                    "(both registers). Each embeds the live spine; view panels backlink here.\n\n"
                    + "\n".join(f"- {N.wl(n.id, n.title)}" for n in notes[:60])
                    + (f"\n- _…{len(notes) - 60} more_" if len(notes) > 60 else ""))
        notes.append(N.Note(id="content/release-chapters", title="Release chapters",
                            kind="content-index", data_product="content", root="current",
                            frontmatter={"n_chapters": len(notes)}, body=idx_body))
    return notes, by_pid


def project_escalation_channel() -> "list[N.Note]":
    """The elaboration escalation channel's health as ONE lineup note (#32): the standing organ's
    KPIs (size, age histogram) + the REVIEW stratum enumerated — aged-out entries are a human
    worklist and must be visible in the lineup, not buried in a build artifact."""
    wl_path = Path("build/elaboration_acp_worklist.json")
    if not wl_path.exists():
        return []
    try:
        wl = json.loads(wl_path.read_text())
    except Exception:  # noqa: BLE001
        return []
    ages: "dict[str, int]" = {}
    review = []
    for key, e in wl.items():
        a = str(e.get("attempts") or 0)
        ages[a] = ages.get(a, 0) + 1
        if e.get("stratum") == "review" or int(e.get("attempts") or 0) >= 3:
            review.append((e.get("template_id") or key.split("::")[0],
                           str(e.get("reasons") or "")[:110]))
    age_line = " · ".join(f"{n}× attempts={a}" for a, n in sorted(ages.items())) or "empty"
    body = [
        f"**{len(wl)} entries** in the standing elaboration worklist "
        f"(never-drop: every membrane run exhales here; the `refine_escalations` flow organ "
        f"consumes it each run). Age histogram: {age_line}.",
        "",
        f"**Review stratum ({len(review)})** — aged past the engine path (≥3 triage passes); "
        "these need a human or a richer proposer:" if review else
        "**Review stratum: empty** — nothing has aged out of the engine path.",
    ]
    body += [f"- `{tid}` — {r}" for tid, r in sorted(review)[:40]]
    return [N.Note(
        id="lexicon/escalation-channel", title="Elaboration escalation channel",
        kind="lexicon-construct", data_product="ontology", root="scratch",
        frontmatter={"lens": "lexicon", "worklist_n": len(wl), "review_n": len(review)},
        body="\n".join(body))]


def project_vocab_concepts(pts: "list[dict]", live_ids: "set | None" = None) -> "list[N.Note]":
    """The vocab's concepts as REAL notes (RH 2026-07-21: constituents were unlinked text — but
    concepts are first-class lexicon citizens; the lens advertises 973 it could not show). One
    note per concept code, carrying its definition text, the anchors that count it a constituent
    (reverse of the snapshot lattice), and a term/class crosslink when resolvable."""
    try:
        from aegir.ontology import domain_index as DI
        vocab = DI.load_skos()
    except Exception:  # noqa: BLE001
        return []
    live_ids = live_ids or set()
    in_anchors: "dict[str, list]" = {}
    for p0 in pts:
        for x in p0.get("constituents") or []:
            if x.get("kind") == "concept" and x.get("code"):
                in_anchors.setdefault(str(x["code"]), []).append(
                    (str(p0.get("id")), p0.get("label") or "", x.get("rel")))
    notes: "list[N.Note]" = []
    for key, c in sorted(vocab.items()):
        code = str(getattr(c, "code", key))
        label = getattr(c, "pref_label", key)
        text = ""
        try:
            text = re.sub(r"\s+", " ", c.text()).strip()[:400]
        except Exception:  # noqa: BLE001
            pass
        camel = re.sub(r"[^A-Za-z0-9]", "", label.title())
        snake = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        cross = ""
        if snake in live_ids:
            cross = f"**Term.** {N.wl(f'ontology/term/{snake}', snake)}\n\n"
        anchors_md = ""
        if code in in_anchors:
            anchors_md = ("**Constituent of.** " + " · ".join(
                N.wl(f"lexicon/aperture/{aid}", albl[:40]) + (f" (rel {rel})" if rel else "")
                for aid, albl, rel in in_anchors[code][:8]) + "\n\n")
        notes.append(N.Note(
            id=f"lexicon/concept/{code}", title=label[:64], kind="lexicon-construct",
            data_product="ontology", root="scratch",
            frontmatter={"code": code},
            links=[f"lexicon/aperture/{a}" for a, _, _ in in_anchors.get(code, [])[:8]],
            body=(f"**{label}** — a vocab concept (`{code}`; the admission surface's unit — "
                  f"see {N.wl('lexicon/construct/concept', 'concept (construct)')}).\n\n"
                  + (f"> {text}\n\n" if text else "") + cross + anchors_md)))
    return notes


def project_aperture_anchors() -> list[N.Note]:
    """The Canonical Aperture's composite anchors as notes (RH 2026-07-19): one per anchor, from the
    strategy lens snapshot (truth flows repo → runtime). Constituent-concept mappings are not yet in
    the snapshot (vector_sha only) — each note says so honestly; surfacing them is #31."""
    try:
        import json as _json
        from aegir.strategy.manifest import read_component
        try:
            pts = _json.loads(read_component("lens/aperture.snapshot.json"))
        except Exception:  # noqa: BLE001 — pre-rename strategy checkouts
            pts = _json.loads(read_component("lens/aiming.snapshot.json"))
        pts = pts.get("points", pts) if isinstance(pts, dict) else pts
    except Exception:  # noqa: BLE001
        return []
    notes = [N.Note(
        id="lexicon/aperture/index", title="Canonical Aperture", kind="lexicon-construct",
        data_product="ontology", root="scratch",
        body=("**The Canonical Aperture** — " + str(len(pts)) + " composite anchors, each compositing "
              "primary constituent concepts (see " + N.wl("lexicon/construct/aperture", "the construct")
              + "). domain⇄concept is many-to-many; anchors are the admission surface.\n\n"
              + "\n".join(f"- {N.wl('lexicon/aperture/' + str(p0.get('id')), p0.get('label') or str(p0.get('id')))}"
                           for p0 in pts)))]
    label_to_aid = {(p0.get("label") or ""): str(p0.get("id")) for p0 in pts}
    for p0 in pts:
        aid, label = str(p0.get("id")), p0.get("label") or str(p0.get("id"))
        notes.append(N.Note(
            id=f"lexicon/aperture/{aid}", title=label, kind="lexicon-construct",
            data_product="ontology", root="scratch",
            frontmatter={"vector_sha": p0.get("vector_sha")},
            body=(f"**{label}** — a Canonical Aperture composite anchor (id `{aid}`, vector "
                  f"`{p0.get('vector_sha')}`).\n\n"
                  + _aperture_constituents_md(p0, label_to_aid)
                  + "Part of " + N.wl("lexicon/aperture/index", "the Canonical Aperture") + " · "
                  + N.wl("lexicon/construct/aperture", "aperture (construct)"))))
    notes += project_vocab_concepts(pts, live_ids=getattr(project_aperture_anchors, "_live_ids", None))
    return notes



def project_shape_surfaces(live_ids: "set | None" = None) -> "tuple[list[N.Note], str]":
    """The UNIVERSAL organizing construct for ontologies under management (RH 2026-07-20): the
    axiom-SHAPE space. Every axiom of ANY OWL ontology classifies into it — including shapes our
    pattern library cannot yet generate — so navigation stops steering users through our (measured
    85.7%-complete) category taxonomy and starts showing each ontology's OWN structure with the
    pattern library as a COVERAGE OVERLAY. Native side = the live catalog classified by the census
    grammar; foreign side = every build/*_coverage.json (foreign_ontology_coverage.py output).
    Returns (notes, the lens lead line)."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("focov", Path("scripts/foreign_ontology_coverage.py"))
    if spec is None or spec.loader is None:
        return [], ""
    focov = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(focov)
        ours, by_pattern = focov.catalog_sigs()
    except Exception:  # noqa: BLE001
        return [], ""
    notes: "list[N.Note]" = []
    # native: shape → the pattern categories that generate it
    shape_pats: "dict[str, list[str]]" = {}
    for pat, sigs_ in by_pattern.items():
        for s_ in sigs_:
            shape_pats.setdefault(s_, []).append(pat)
    rows = ["| axiom shape | generated by |", "|---|---|"]
    for s_ in sorted(shape_pats):
        pats = sorted(shape_pats[s_])
        cell = " · ".join(N.wl(f"ontology/category/{c}", c) if c not in
                          ("(uncategorized)", "(realizer-machinery)") else f"_{c}_"
                          for c in pats[:6]) + (f" _…+{len(pats) - 6}_" if len(pats) > 6 else "")
        rows.append(f"| `{s_}` | {cell} |")
    notes.append(N.Note(
        id="ontology/shapes", title="Axiom shapes — the universal organizer",
        kind="lexicon-construct", data_product="ontology", root="scratch",
        frontmatter={"n_shapes": len(shape_pats)},
        body=("**Axiom shapes.** The ontology-agnostic structural space every OWL axiom "
              "classifies into — the lineup's organizing construct for ontologies under "
              "management. The pattern-category library is a COVERAGE OVERLAY on this space "
              "(what we can *generate*), never the navigation itself: a foreign ontology's "
              "shapes outside our envelope stay visible as first-class gaps.\n\n"
              + "\n".join(rows) + "\n")))
    # foreign ontologies under management: one panel per census artifact
    fentries = []
    live_ids = live_ids or set()

    def _snake(x: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", x).replace("-", "_").lower()

    for cov_p in sorted(Path("build").glob("*_coverage.json")):
        try:
            cov = json.loads(cov_p.read_text())
        except Exception:  # noqa: BLE001
            continue
        tag = cov.get("tag", cov_p.stem.replace("_coverage", ""))
        native = tag == "sdg"
        base = "ontology/self-census" if native else f"ontology/foreign/{tag}"
        tb_tot, tb_cov = cov.get("tbox_rbox_total", 0), cov.get("tbox_rbox_covered", 0)
        pct = 100 * tb_cov / max(1, tb_tot)
        blocked = cov.get("blocked", {})
        covered = cov.get("covered", {})
        ex = cov.get("blocked_examples", {})
        pw = cov.get("pathways") or {}
        groups_ = pw.get("groups") or []
        edges_ = pw.get("edges") or []
        # local class name → its group slug (for example-linking on foreign ontologies)
        local_slug = {m.get("local"): g["slug"] for g in groups_ for m in g.get("members", [])}

        def _ex_cell(e0) -> str:
            if isinstance(e0, str):                      # pre-upgrade artifact
                return e0[:60]
            loc = e0.get("local", "")
            if native and _snake(loc) in live_ids:
                return N.wl(f"ontology/term/{_snake(loc)}", loc[:46])
            if loc in local_slug:
                return N.wl(f"{base}/group/{local_slug[loc]}", loc[:46])
            return loc[:60]

        brows = ["| blocked shape | n | example |", "|---|---|---|"]
        for k, v in list(blocked.items())[:16]:
            if k.startswith("abox:"):
                continue
            brows.append(f"| {N.wl('ontology/shapes', k)} | {v} | {_ex_cell((ex.get(k) or [''])[0])} |")
        crows = ["| covered shape | n |", "|---|---|"]
        for k, v in list(covered.items())[:10]:
            crows.append(f"| {N.wl('ontology/shapes', k)} | {v} |")
        grp_line = " · ".join(
            N.wl(f"{base}/group/{g['slug']}", f"{g['group'][:28]} ({g['n_classes']})")
            for g in groups_[:14]) or "—"
        mods = sorted((cov.get("per_module") or {}).items(), key=lambda kv: -kv[1])[:8]
        opa = blocked.get("abox:opa", 0)
        notes.append(N.Note(
            id=base, title=("SDG — native self-census" if native else
                            f"{tag.upper()} — foreign ontology under management"),
            kind="lexicon-construct", data_product="ontology", root="scratch",
            frontmatter={"tag": tag, "tbox_coverage": round(pct, 1),
                         "n_logical": cov.get("n_logical_axioms"),
                         "viz_view": "pathways", "viz_onto": tag},
            links=[f"{base}/group/{g['slug']}" for g in groups_[:14]] + ["ontology/shapes"],
            body=(f"**{tag.upper()}** under management — censused by the shape grammar "
                  f"({cov.get('n_logical_axioms', 0):,} logical axioms).\n\n"
                  f"**Groups (the chord's arcs — click through).** {grp_line}\n\n"
                  f"**Schema expressibility: {tb_cov:,}/{tb_tot:,} = {pct:.1f}%** generable by the "
                  "current pattern library; the remainder are first-class GAPS (pattern-mint work, "
                  "not invisible residue).\n\n"
                  + "\n".join(crows) + "\n\n" + "\n".join(brows) + "\n\n"
                  + (f"**Reference-data layer.** {opa:,} individual-to-individual assertions — "
                     "our architecture routes this layer to the RELATIONAL product with lineage, "
                     "not OWL ABox.\n\n" if opa else "")
                  + "**Module loads.** " + " · ".join(f"`{m}` {n}" for m, n in mods) + "\n\n"
                  + f"_Census artifact: `build/{tag}_coverage.json` · re-run: "
                  f"`scripts/foreign_ontology_coverage.py --root <checkout> --tag {tag}`_\n")))
        # ── group panels: the NAVIGABLE pathways at panel grain ──
        for g in groups_:
            slug = g["slug"]
            out_e = [e for e in edges_ if e.get("src_slug") == slug][:10]
            in_e = [e for e in edges_ if e.get("dst_slug") == slug and e.get("src_slug") != slug][:8]
            gslugs = {g_["slug"] for g_ in groups_}

            def _gcell(sl: str, label: str, n_: int) -> str:
                return (N.wl(f"{base}/group/{sl}", label[:34]) + f" ({n_})") if sl in gslugs \
                    else f"`{label[:34]}` ({n_})"
            mem_cells = []
            for m in g.get("members", [])[:40]:
                loc = m.get("local", "")
                if native and _snake(loc) in live_ids:
                    mem_cells.append(N.wl(f"ontology/term/{_snake(loc)}", m.get("label", loc)[:34]))
                else:
                    mem_cells.append(N.wl(f"{base}/class/{loc}", m.get("label", loc)[:34]))
            body_g = (
                f"**{g['group']}** — a pathway group of {N.wl(base, tag.upper())} "
                f"({g['n_classes']} classes; the chord arc).\n\n"
                + "**Outbound.** " + ("\n".join(
                    f"- →[`{' · '.join(e.get('props', [])[:3])}`] "
                    + _gcell(e["dst_slug"], e["dst"], e["n"])
                    for e in out_e) or "—") + "\n\n"
                + "**Inbound.** " + (" · ".join(
                    _gcell(e["src_slug"], e["src"], e["n"]) for e in in_e) or "—") + "\n\n"
                + "**Members.** " + (" · ".join(mem_cells) or "—")
                + (f" _…+{g['n_classes'] - len(mem_cells)}_" if g["n_classes"] > len(mem_cells) else "")
                + "\n")
            notes.append(N.Note(
                id=f"{base}/group/{slug}", title=f"{g['group'][:52]} · {tag}",
                kind="lexicon-construct", data_product="ontology", root="scratch",
                links=[base] + [f"{base}/group/{e['dst_slug']}" for e in out_e],
                frontmatter={"tag": tag, "n_classes": g["n_classes"]},
                body=body_g))
        if not native:
            fentries.append((base, tag, pct))
    lens_line = ("**Under management.** " + N.wl("ontology/shapes", "axiom shapes (universal)")
                 + " organize every ontology here — ours and foreign: native "
                 + N.wl("ontology/self-census", "sdg")
                 + ("".join(f" · {N.wl(nid, tag.upper())} ({pct:.0f}% expressible)"
                            for nid, tag, pct in fentries))
                 + ". The pattern categories are the GENERATIVE COVERAGE overlay on that space, "
                   "not the organizer.\n\n")
    return notes, lens_line


def project_trunk_lenses(categories: list[str], rel_cats: list[str], has_sdg: bool,
                         zettel_head: str | None = None,
                         items_report: dict | None = None,
                         shapes_line: str = "") -> list[N.Note]:
    """The TRUNK kasten's lenses (scratch root) — the same lens ids as the release kasten,
    root-resolved (roots are refs, git-style). These browse the LIVE state: the derived
    catalog by pattern, the grounds-shape spine + the earned generated web, and the
    accreting corpus. ``chord: false`` — the collection chords are release-era."""
    # Aperture counts read THROUGH the strategy (truth flows repo → runtime); absent → omitted.
    ap_line = ""
    try:
        import json as _json
        from aegir.strategy.manifest import read_component
        vocab = _json.loads(read_component("lens/vocab.snapshot.json"))
        try:
            aiming = _json.loads(read_component("lens/aperture.snapshot.json"))
        except Exception:  # noqa: BLE001 — pre-rename strategy checkouts
            aiming = _json.loads(read_component("lens/aiming.snapshot.json"))
        n_c = len(vocab.get("points", vocab) if isinstance(vocab, dict) else vocab)
        n_d = len(aiming.get("points", aiming) if isinstance(aiming, dict) else aiming)
        ap_line = (f"**Canonical Aperture.** {n_d} composite anchors (each compositing primary "
                   f"constituent concepts) · {n_c} concepts (vocab) — the admission surface, pinned "
                   f"by the strategy lens pillar. domain⇄concept is MANY-TO-MANY (faceted navigation, "
                   f"never a tree): " + N.wl("lexicon/aperture/index", "browse the anchors") + ".\n\n")
    except Exception:  # noqa: BLE001 — strategy snapshots optional at build time
        ap_line = ("**Aperture.** domains (aiming) + concepts (vocab) — the admission surface, "
                   "pinned by the strategy lens pillar (snapshots not readable at build).\n\n")
    terms = N.Note(
        id="lens/terms", title="Lexicon (trunk)", kind="lens", data_product="ontology",
        root="scratch", frontmatter={"lens": "terms", "lexicon": LEXICON, "chord": False,
                                     "viz_view": "pathways", "viz_onto": "sdg"},
        body=("*The chord above is the ontology's INHERENT navigable structure — its own "
              "discriminating ancestors, its restriction web and property lattice. Universally "
              "available for any ontology under management; maxsim/aperture is ONE further view, "
              "not the organizer.*\n\n"
              + shapes_line
              + ap_line
              + "**Generative coverage.** The live catalog by pattern category (the overlay, "
                "demonstrably incomplete against foreign ontologies): "
              + " · ".join(N.wl(f"ontology/category/{c}", c) for c in categories[:10])
              + (f" _…+{len(categories) - 10}_" if len(categories) > 10 else "") + "\n\n"
              + "**Topics.** " + N.wl("topic/index", "anchor-topics") + " (inverted layer; "
                "latent-topics staged #30) · **Collections.** " + N.wl("collection/index", "index")
              + " · **Constructs.** "
              + " · ".join(N.wl(f"lexicon/construct/{c}", c) for c in
                           ("domain", "concept", "topic", "collection", "term", "genus",
                            "differentia")) + "\n"))
    schema_body = ("**Schema (trunk).** The live relational surfaces.\n")
    if has_sdg:
        schema_body += (f"\n**The generated web (earned).** {N.wl('relational/sdg-schema', 'SDG schema')} — "
                        "the `just metaflow` constructs, verbatim, with their earned cross-entity FK edges.\n")
    schema_body += ("\n**The deterministic spine, by grounds-shape:**\n\n"
                    + "\n".join(f"- {N.wl(f'relational/category/{c}', c)}" for c in rel_cats))
    schema = N.Note(
        id="lens/schema", title="Schema (trunk)", kind="lens", data_product="relational",
        root="scratch", frontmatter={"lens": "schema", "lexicon": LEXICON, "chord": False},
        body=schema_body)
    content_body = ("**Content × Topics (trunk).** The accreting corpus, pivoted over the "
                    "INVERTED TOPIC LAYER — term-grounded topics and the items that "
                    "unambiguously bind to them (margin-gated; the lens shape is invariant "
                    "across roots, the substrate is era-true).\n\n"
                    f"- {N.wl('corpus/sdg', 'the live SDG corpus')}\n")
    if zettel_head:
        content_body += f"- run chain head: {N.wl('corpus/runs/' + zettel_head, zettel_head)}\n"
    chord_on = False
    if items_report:
        content_body += (f"- {N.wl('item/index', 'Items × Topics index')} — "
                         f"{items_report.get('n_assigned')}/{items_report.get('n')} items aligned "
                         f"to {items_report.get('topics_hit')} topics "
                         f"(registry `@{items_report.get('collection_sha')}`)\n")
        by_topic = items_report.get("by_topic") or {}
        if by_topic:
            chord_on = True
            rows = ["", "| topic | aligned items |", "|---|---|"]
            for key, n_items in list(by_topic.items())[:40]:
                code = key.split(" ")[0]
                cell = (N.wl(f"ontology/term/{code}", code) if not code[0].isdigit()
                        else f"`{code}` {key[len(code):].strip()} (domain)")
                rows.append(f"| {cell} | {n_items} |")
            content_body += "\n".join(rows) + "\n"
    content = N.Note(
        id="lens/content", title="Content × Topics (trunk)", kind="lens", data_product="content",
        root="scratch", frontmatter={"lens": "content", "chord": chord_on}, body=content_body)
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
        if cslug == "provenance-authenticity":
            man = S.spine_manifest()
            rs = (man or {}).get("realize_summary") or {}
            psd = rs.get("profile_source_distribution") or {}
            if man and psd:
                signal = sum(v for k, v in psd.items() if k.startswith("grounds_ddl:"))
                static = sum(v for k, v in psd.items() if not k.startswith("grounds_ddl:"))
                vps = rs.get("value_pool_sources") or {}
                cat_fm = {"realize_summary": rs}
                live_block = (
                    f"\n\n**LIVE** (spine `{man['_run']}`): profile sources — signal-driven "
                    f"**{signal}** vs default-minimal **{static}** (static fraction "
                    f"**{static / (signal + static):.1%}**, dial → 0; the default-minimal are the "
                    f"intermediate classes, which carry no grounding signal) · value pools "
                    f"{' · '.join(f'{k} {v}' for k, v in sorted(vps.items()))} · "
                    f"{rs.get('tables_per_template')} tables/template · "
                    f"FK depth ≤ {rs.get('max_fk_depth')}\n")
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


_ISO_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def run(args=None) -> int:
    kb = S.kb_dir()
    current = kb / "current"
    shutil.rmtree(current, ignore_errors=True)        # regenerable projection
    for d in (current, kb / "scratch", kb / "archive"):
        d.mkdir(parents=True, exist_ok=True)
    # Projector-owned cleanup (roots are refs, RH 2026-07-06: current = the latest release
    # kasten · scratch = TRUNK, the live projection · archive = past releases + snapshots).
    # scratch: everything except authored <iso-date>/ dirs is projected trunk state.
    # archive: the projected subtrees; snapshots (<year>Q<n>*) + aged authored notes stay.
    for child in (kb / "scratch").iterdir():
        if child.is_dir() and _ISO_DIR.match(child.name):
            continue
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    for sub in ("ontology", "relational", "release", "corpus"):
        shutil.rmtree(kb / "archive" / sub, ignore_errors=True)

    rows = S.load_ontology()
    categories = sorted({cat for cat, _ in rows})          # axiom-pattern axis (+ intermediate)
    tid2cat = {t.template_id: cat for cat, t in rows}
    rel_cats = sorted({S.relational_category(t) for _, t in rows})   # grounds-shape axis
    sc = S.sdg_constructs()   # the generated web — read early so downstream surfaces can link it
    rd = S.released_ddl()     # the RELEASED DDL spine (corpora/ddl) — current-root schema surface
    rels = S.releases()       # the sdg-corpora release records — read early so lenses can link them
    latest_rel = rels[-1] if rels else None
    # The superseded catalog generations (git-recovered). Era-alignment (RH): the ids the
    # RELEASED corpus cites are the release's lexicon → current citizens; uncited → archive.
    retired = [(f, t) for f, t in S.load_retired_ontology() if t.template_id not in tid2cat]
    retired_cat = {t.template_id: f for f, t in retired}

    # Read content/coverage once + derive cross-references (so Terms show what exercises them).
    corpus, crun = S.corpus_recs()
    coverage, cov = S.coverage_recs()
    term_topics = assign_topic_threads(corpus, coverage)
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
    cgraph: dict = {}
    if sc:
        h6_tables: dict = {}
        table_cols: dict = {}
        for tn, te in (sc.get("tables") or {}).items():
            table_cols[tn] = [(c["name"], c.get("concept")) for c in te.get("columns", [])]
            for pid in te.get("constructs", []):
                h6_tables.setdefault(pid[:6], set()).add(tn)
        view_pids: dict = {}
        for vn, ve in (sc.get("views") or {}).items():
            for pid in (ve.get("constructs") or [ve.get("construct")]):
                if pid:
                    view_pids.setdefault(vn, set()).add(pid)
        cgraph = {"h6_tables": h6_tables, "table_cols": table_cols, "view_pids": view_pids,
                  "term_elements": {t.template_id: sorted((t.slot_types or {}).keys())
                                    for _, t in rows}}
    # era-correct archive targets (RH: link archive resources, never dead-end on 'retired'):
    # the NEWEST archived kasten's term panels — the term as it stood in that freeze.
    arch_root = Path(S.kb_dir()) / "archive"
    kastens = sorted(arch_root.glob("*/*/ontology/term"), key=lambda d: d.as_posix())
    if kastens:
        kd = kastens[-1]
        prefix = kd.parent.parent.relative_to(arch_root).as_posix()
        at: dict = {}
        for f in kd.glob("*.md"):
            try:
                meta, _b = N.parse_markdown(f.read_text())
            except Exception:  # noqa: BLE001
                meta = {}
            at[f.stem] = {"id": f"{prefix}/ontology/term/{f.stem}",
                          "slots": sorted((meta.get("slot_types") or {}).keys())}
        cgraph["archived_terms"] = at
    if corpus and coverage:
        coll_notes, maps = project_collections(corpus, coverage, live_terms=set(tid2cat),
                                               cgraph=cgraph)
    rel_ch_notes, rel_by_pid = project_release_chapters(sc)
    rel_coll_notes, rel_table_colls = project_release_collections(
        sc, rel_by_pid, live_ids={t.template_id for _, t in rows})
    for t_, cids_ in rel_table_colls.items():
        maps.setdefault("table_colls", {}).setdefault(t_, []).extend(cids_)

    # TRUNK (scratch): the live catalog's lexicon + its deterministic spine — where new
    # work lands, git-trunk-style. The whole live projection is a scratch citizen.
    assoc = S.topic_associations()
    item_notes: list[N.Note] = []
    term_items: dict[str, list[str]] = {}
    if assoc:
        item_notes, term_items = project_items(assoc, live_terms=set(tid2cat))
        print(f"  items: {len(item_notes) - 1} aligned passage windows "
              f"(inverted layer, registry @{assoc['report'].get('collection_sha')}, "
              f"rate {assoc['report'].get('alignment_rate')})")
    trunk = (project_ontology(rows, term_chapters, term_topics, broader, narrower,
                              term_colls=maps.get("term_colls"), term_items=term_items)
             + project_relational(rows, has_sdg=bool(sc))
             + item_notes)
    for n in trunk:
        n.root = "scratch"
    notes = trunk
    print(f"  trunk(scratch): {len(trunk)} notes from {len(rows)} live terms "
          f"in {len(categories)} pattern categories × {len(rel_cats)} grounds shapes "
          f"(Lexicon {LEXICON!r})")
    # RELEASE ERA (current/archive): the generation the released corpus cites is the
    # release's lexicon (current citizens); recovered-but-uncited ids → archive tombstones.
    # RH 2026-07-19: "cited by the release" means cited by the v0.6 RELEASE — the entities its
    # constructs realize — not the path-a trunk corpus (the old input let retired-generation
    # templates masquerade as current citizens: old templates in front of the new structure).
    cited: set[str] = set()
    if sc:
        for tn, te in (sc.get("tables") or {}).items():
            cited.add(tn)
            cited |= {c.get("concept") for c in te.get("columns", []) if c.get("concept")}
    if not cited:                       # no release run on disk → fall back to the trunk corpus
        for r in corpus:
            cited |= {str(x) for x in (r.get("template_ids") or [])}
        cited |= {str(r["top_template_id"]) for r in coverage if r.get("top_template_id")}
    if retired:
        era_tables = {r["template_id"]: r["table_name"] for r in (rd or {}).get("rows", [])}
        rt = project_retired_ontology(retired, term_chapters, term_topics,
                                      term_colls=maps.get("term_colls"),
                                      cited=cited, era_tables=era_tables)
        notes += rt
        n_cur = sum(1 for n in rt if n.root == "current")
        print(f"  release lexicon: {len(rt)} era notes ({n_cur} current citizens — the released "
              f"corpus's terms; {len(rt) - n_cur} uncited → archive)")
    if rd:
        rdn = project_released_ddl(rd)
        notes += rdn
        print(f"  release ddl: {len(rdn)} current notes from corpora/ddl `{rd['run']}` "
              f"({len(rd['rows'])} released tables)")
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
        tp = project_topics(coverage, topic_chapters, topic_colls=maps.get("topic_colls"),
                            tid2cat=tid2cat, retired_cat=retired_cat)
        notes += tp
        print(f"  topics:  {len(tp)} notes   ({cov})")
    else:
        print("  topics:  (no on-disk coverage run — skipped; set AEGIR_COVERAGE_RUN to project)")

    if coll_notes:
        notes += coll_notes
        print(f"  collections: {len(coll_notes) - 1} topic-grounded bundles (many-to-many: "
              f"{len(maps.get('topic_colls', {}))} topics × {len(maps.get('term_colls', {}))} terms)")

    era_fams = sorted({f for f, _ in retired})
    _rc = [(n_.id, n_.frontmatter.get("label", n_.title), n_.frontmatter.get("n_chapters", 0))
           for n_ in rel_coll_notes if n_.kind == "collection"]
    _rs: dict = {}
    for t_, cids_ in rel_table_colls.items():
        for c_ in cids_:
            _rs.setdefault(c_, []).append(t_)
    notes += project_lenses(era_fams, bool(corpus), bool(coverage), maps, rd=rd, rel_colls=_rc,
                            rel_schema=_rs,
                            live_terms=set(tid2cat),
                            latest_release=latest_rel["version"] if latest_rel else None)
    tr = project_training() + project_metrics()
    notes += tr
    # TRAINING twins on trunk (RH 2026-07-06) — the procedures run against the in-progress
    # corpora as we iterate toward the next release, so the instruments live in scratch too.
    import copy
    trunk_tr = [copy.deepcopy(n) for n in tr]
    for n in trunk_tr:
        n.root = "scratch"
    notes += trunk_tr
    print(f"  training: 3 viz panels + metrics catalog ({len(tr)} notes, twinned current+scratch)")

    strat_id = None
    st = S.strategy_state()
    if st:
        man, pillars = st["manifest"], st["manifest"]["pillars"]
        strat_id = man["strategy_id"]
        pl_lines = []
        for pil, comps in pillars.items():
            pl_lines.append(f"\n**{pil}**")
            pl_lines += [f"- `{c}` — `{man['components'][c][:16]}`" for c in comps]
        notes.append(N.Note(
            id=f"strategy/{man['strategy_id']}", title=f"strategy {man['strategy_id']}",
            kind="strategy", data_product="strategy", root="scratch",
            frontmatter={"strategy_id": man["strategy_id"], "commit": st["commit"],
                         "n_components": len(man["components"])},
            body=(f"**Strategy `{man['strategy_id']}`** @ sdg-strategy `{st['commit']}` — the "
                  f"run's externalized determinants, content-addressed (identity = the root "
                  f"hash; verifiable by rehash from any signals project). Four pillars, "
                  f"{len(man['components'])} components, VERBATIM:\n"
                  + "\n".join(pl_lines))))
        print(f"  strategy: {man['strategy_id']} @ {st['commit']} "
              f"({len(man['components'])} components)", flush=True)

    if sc:
        tbls, vws = sc["tables"], sc["views"]
        known = set(tbls)
        t_lines = [f"- {N.wl('relational/table/' + n, n)} — {len(e['columns'])} cols · "
                   f"{len(e['fks'])} FKs · pk {e.get('pk_kind') or e.get('pk') or '—'}"
                   for n, e in sorted(tbls.items())]
        notes.append(N.Note(
            id="relational/sdg-schema", title="SDG schema (generated)", kind="relational",
            data_product="relational", root="scratch",
            body=(f"**The generated relational product, verbatim** — {len(tbls)} unique tables "
                  f"+ {len(vws)} views across {sc['n_constructs']} constructs (`just metaflow` "
                  f"corpus). Table and column names are the artifacts themselves — no wrappers.\n\n"
                  + "\n".join(t_lines))))
        # shape-bound DOMAINS from the run's SHACL shapes (the differentia arc, visible in the panel):
        # class → table via the register projections; enum-kind props → `col ∈ {…}` constraint lines.
        shape_proj: dict = {}
        shape_idx: dict = {}
        try:
            from aegir.lineup.sources import sdg_run_root
            from aegir.ontology.discriminability import shape_projections
            from aegir.ontology.entities import _plural, _snake
            _shp = (sdg_run_root() or Path("/nonexistent")) / "ontology" / "shapes.ttl"
            if _shp.exists():
                shape_proj = shape_projections(_shp)
                for _cls in shape_proj:
                    for _key in (_cls, _snake(_cls), _plural(_snake(_cls))):
                        shape_idx.setdefault(_key, _cls)
        except Exception as _e:  # noqa: BLE001 — shapes are additive; the panel renders without them
            print(f"  (shape-bound constraints unavailable: {str(_e)[:80]})", flush=True)
        for n, e in tbls.items():
            # CONSTRAINTS — the unbounded category gets its own SECTION (RH): full statements live here
            # (PRIMARY KEY · FOREIGN KEY · shape-bound enum domains, more kinds as they become
            # derivable); the schema table keeps `key` as the bounded at-a-glance marker.
            pk_cols = e.get("pk") if isinstance(e.get("pk"), list) else ([e.get("pk")] if e.get("pk") else [])
            fk_lines = ([f"- `PRIMARY KEY ({', '.join(pk_cols)})`"] if pk_cols else [])
            fk_lines += [f"- `FOREIGN KEY {fk.get('col')}` → " +
                         (N.wl("relational/table/" + fk.get("ref_table", ""), fk.get("ref_table", ""))
                          if fk.get("ref_table") in known else f"`{fk.get('ref_table')}`") +
                         f" · `{fk.get('ref_col')}`"
                         for fk in e["fks"]]
            _cls = shape_idx.get(n)
            if _cls:
                from aegir.ontology.entities import _snake as _sn
                colnames = {c["name"] for c in e["columns"]}
                snake_reg = any("_" in c["name"] for c in e["columns"]) or n == n.lower()
                for _prop, (_kind, _val) in sorted(shape_proj.get(_cls, {}).items()):
                    if _kind != "enum":
                        continue
                    _col = _sn(_prop) if snake_reg else _prop
                    if _col in colnames:
                        fk_lines.append(f"- `{_col} ∈ {{{' · '.join(sorted(_val))}}}` "
                                        f"— shape-bound domain (sh:in)")
            prov = ", ".join(e["constructs"][:4]) + ("…" if len(e["constructs"]) > 4 else "")
            # the column detail TABLE — schema only; `key` stays the bounded at-a-glance marker, the
            # full (unbounded) constraint statements render in the Constraints section below.
            col_rows = ["| column | type | key |", "|---|---|---|"]
            for c in e["columns"]:
                key = ("PK" if c["pk"] else "") + ("+" if c["pk"] and c["fk"] else "") + \
                      (f"FK → {N.wl('relational/table/' + c['fk'].split('.')[0], c['fk'])}"
                       if c["fk"] and c["fk"].split(".")[0] in known
                       else (f"FK → `{c['fk']}`" if c["fk"] else ""))
                col_rows.append(f"| `{c['name']}` | {c['type']} | {key or '—'} |")
            ref_by = [f"- {N.wl('relational/table/' + r['table'], r['table'])} · `{r['col']}`"
                      for r in e.get("referenced_by", [])[:10] if r["table"] in known]
            more_ref = len(e.get("referenced_by", [])) - len(ref_by)
            vw = e.get("views", [])
            vw_lines = [f"- {N.wl('relational/view/' + v, v)}"
                        + (f" ({vws[v].get('kind')})" if vws.get(v, {}).get("kind") else "")
                        for v in vw[:8]]
            sample_tables = []
            for c in e["columns"]:
                if c["samples"]:
                    sample_tables.append(f"| `{c['name']}` |\n|---|\n"
                                         + "\n".join(f"| {s} |" for s in c["samples"]))
            # the first-order ERD payload (rendered by the React panel; bounded neighborhood)
            erd_nodes = [{"id": n, "kind": "junction" if e.get("junction") else "table", "focal": True,
                          "cols": [f"{c['name']}: {c['type']}" for c in e["columns"][:8]]}]
            erd_edges = []
            for fk in e["fks"][:6]:
                rt = fk.get("ref_table")
                if rt and rt != n:
                    erd_nodes.append({"id": rt, "kind": "junction" if tbls.get(rt, {}).get("junction")
                                      else "table", "known": rt in known})
                    erd_edges.append({"source": n, "target": rt, "label": fk.get("col")})
            for r in e.get("referenced_by", [])[:6]:
                if r["table"] != n and not any(x["id"] == r["table"] for x in erd_nodes):
                    erd_nodes.append({"id": r["table"], "kind": "junction"
                                      if tbls.get(r["table"], {}).get("junction") else "table",
                                      "known": r["table"] in known})
                    erd_edges.append({"source": r["table"], "target": n, "label": r["col"]})
            for v in vw[:4]:
                erd_nodes.append({"id": v, "kind": "view"})
                erd_edges.append({"source": n, "target": v, "label": "view"})
            notes.append(N.Note(
                id=f"relational/table/{n}", title=n, kind="relational-table",
                data_product="relational", root="scratch",
                frontmatter={"pk": e.get("pk"), "pk_kind": e.get("pk_kind"),
                             "n_columns": len(e["columns"]), "constructs": len(e["constructs"]),
                             "junction": bool(e.get("junction")),
                             "erd": {"focal": n, "nodes": erd_nodes, "edges": erd_edges,
                                     "more": {"referenced_by": max(0, len(e.get("referenced_by", [])) - 6),
                                              "views": max(0, len(vw) - 4)}}},
                links=[f"relational/table/{fk.get('ref_table')}" for fk in e["fks"]
                       if fk.get("ref_table") in known],
                body=(f"**`{n}`** — generated {'junction ' if e.get('junction') else ''}table "
                      f"(verbatim; pk kind **{e.get('pk_kind') or ('composite' if e.get('junction') else '—')}**, "
                      f"pk `{e.get('pk') or '—'}`).\n\n"
                      + "\n".join(col_rows) + "\n\n"
                      + ("**Constraints**\n" + "\n".join(fk_lines) + "\n\n" if fk_lines else "")
                      + ("**Referenced by**\n" + "\n".join(ref_by)
                         + (f"\n- _…{more_ref} more_" if more_ref > 0 else "") + "\n\n" if ref_by else "")
                      + ((f"**Views over this table** ({len(vw)})\n" + "\n".join(vw_lines)
                          + (f"\n- _…{len(vw) - 8} more_" if len(vw) > 8 else "") + "\n\n") if vw else "")
                      + (("**Sample Values**\n\n" + "\n\n".join(sample_tables) + "\n\n")
                         if sample_tables else "")
                      + (("**Collections** (" + str(len(maps.get("table_colls", {}).get(n, []))) + ")\n"
                          + "\n".join(f"- {N.wl(c + '/terms', c.split('/')[-1] + ' · terms')}"
                                       for c in maps.get("table_colls", {}).get(n, [])[:10]) + "\n\n")
                         if maps.get("table_colls", {}).get(n) else "")
                      + f"_Constructs: {prov}_")))
        # VIEW notes (RH 2026-07-18): every view is a first-class panel with the same anatomy as its
        # source tables — ERD header (source tables → the view), output columns (from the view's own
        # SELECT list — the constructs' columns field under-reports), RESULT samples, and the SQL.
        from aegir.lineup.sources import view_select_cols
        for vn, v in vws.items():
            srcs = [t for t in v.get("tables", []) if t in known]
            headers = view_select_cols(v.get("sql", "")) or v.get("columns", [])
            rows_ = v.get("rows") or []
            width = max((len(r) for r in rows_), default=len(headers))
            if len(headers) < width:
                headers = headers + [f"col{i + 1}" for i in range(len(headers), width)]
            headers = headers[:10]
            vtypes = []
            for i in range(len(headers)):
                col_vals = [str(r[i]) for r in rows_ if i < len(r)]
                vtypes.append(S._infer_sql_type(col_vals) if col_vals else "text")
            col_tbl = ["| column | type |", "|---|---|"] + \
                      [f"| `{h}` | {t} |" for h, t in zip(headers, vtypes)]
            sample_tbls = []
            for i, h in enumerate(headers):
                vals = [str(r[i])[:28] for r in rows_[:4] if i < len(r)]
                if vals:
                    sample_tbls.append(f"| `{h}` |\n|---|\n" + "\n".join(f"| {v} |" for v in vals))
            erd_nodes = [{"id": vn, "kind": "view", "focal": True,
                          "cols": [f"{h}: {t}" for h, t in list(zip(headers, vtypes))[:8]]}]
            erd_edges = []
            for t in srcs[:6]:
                erd_nodes.append({"id": t, "kind": "junction" if tbls.get(t, {}).get("junction")
                                  else "table", "known": True})
                erd_edges.append({"source": t, "target": vn, "label": ""})
            notes.append(N.Note(
                id=f"relational/view/{vn}", title=vn, kind="relational-view",
                data_product="relational", root="scratch",
                frontmatter={"view_kind": v.get("kind"), "n_columns": len(headers),
                             "erd": {"focal": vn, "nodes": erd_nodes, "edges": erd_edges,
                                     "more": {"referenced_by": max(0, len(srcs) - 6), "views": 0}}},
                links=[f"relational/table/{t}" for t in srcs],
                body=(f"**`{vn}`** — generated view (verbatim; kind **{v.get('kind') or '—'}**) over "
                      + (" · ".join(N.wl("relational/table/" + t, t) for t in srcs) or "`—`") + ".\n\n"
                      + "\n".join(col_tbl) + "\n\n"
                      + (("**Sample Values**\n\n" + "\n\n".join(sample_tbls) + "\n\n")
                         if sample_tbls else "")
                      + "```sql\n" + (v.get("sql") or "").strip() + "\n```\n\n"
                      + (("**Embedded in documents**\n" + "\n".join(
                            f"- {N.wl(c, pid_[:12] + ' · ' + c.rsplit('.', 1)[-1])}"
                            for pid_ in sorted(cgraph.get("view_pids", {}).get(vn, ()))
                            for c in rel_by_pid.get(pid_, [])[:2]) + "\n\n")
                         if any(rel_by_pid.get(p_) for p_ in cgraph.get("view_pids", {}).get(vn, ()))
                         else "")
                      + f"_Construct: {v.get('construct')}_")))
        print(f"  relational(sdg): {len(tbls)} tables + {len(vws)} views VERBATIM "
              f"from {sc['n_constructs']} constructs", flush=True)

    # Released corpus generations (the sdg-corpora CARDs) — the release records:
    # the LATEST release is a current-root citizen, past releases are archive citizens.
    for rel in rels:
        latest = rel is latest_rel
        others = " · ".join(N.wl(f"release/v{r['version']}", f"v{r['version']}")
                            for r in rels if r is not rel) or "—"
        notes.append(N.Note(
            id=f"release/v{rel['version']}",
            title=f"corpus release v{rel['version']}" + (" (latest)" if latest else ""),
            kind="release-note", data_product="corpus",
            root="current" if latest else "archive",
            frontmatter={"version": rel["version"], "latest": latest, "source": rel["path"]},
            body=(f"**Released corpus generation v{rel['version']}**"
                  f"{' — the LATEST release (current citizen)' if latest else ' — a PAST release (archive citizen)'}."
                  f" The SHARE record: `{rel['path']}` (sdg-corpora). Other releases: {others}. "
                  f"The unreleased accretion on top of this lives in `scratch` — "
                  + N.wl("corpus/sdg", "the live corpus") + ".\n\n---\n\n" + rel["body"])))
    if rels:
        print(f"  releases: {len(rels)} corpus cards (latest v{rels[-1]['version']} → current, "
              f"{len(rels) - 1} past → archive)")

    gc = S.sdg_corpus()
    zs: list = []
    if gc:
        from aegir.lineup.zettel import chain_roots, run_zettels
        zs = run_zettels(Path(gc["root"]))
        zroots = chain_roots(zs)
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
        chain_line = ""
        if zs:
            head = zs[-1]
            chain_line = (f"\n\n**Run chain** ({len(zs)} zettels; the unreleased accretion lives in "
                          f"`scratch` — #147 lifecycle): head {N.wl('corpus/runs/' + head['id'], head['id'])}")
        if latest_rel:
            chain_line += (f"\n**Latest release.** "
                           + N.wl(f"release/v{latest_rel['version']}", f"v{latest_rel['version']}")
                           + " (past releases live in `archive`)")
        if strat_id:
            chain_line += f"\n**Strategy.** {N.wl(f'strategy/{strat_id}', strat_id)} (the run's externalized determinants)"
        notes.append(N.Note(
            id="corpus/sdg", title="SDG corpus (live)", kind="corpus",
            data_product="corpus", root="scratch",
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
                  + cg_line + chain_line +
                  f"\n\nSource: `{gc['root']}` (metrics.json · congruence.json · concept_graph.json)")))
        print(f"  corpus: sdg live note ({gc['passages_derived']} passages, "
              f"{gc['chapters_natural'] + gc['chapters_semantic']} chapters)")
        for z in zs:
            w, zst = z.get("window") or {}, z.get("state") or {}
            zv = z.get("versions") or {}
            zstrat = zv.get("strategy_id")
            strat_line = ""
            if zstrat:
                strat_line = ("\n- strategy: " + (N.wl(f"strategy/{zstrat}", zstrat)
                                                  if zstrat == strat_id else f"`{zstrat}`"))
            notes.append(N.Note(
                id=f"corpus/runs/{z['id']}", title=z["id"], kind="corpus-run",
                data_product="corpus", root=zroots.get(z["id"], "scratch"),
                frontmatter={"prev": z.get("prev"), "at": z.get("at"),
                             "deriver": zv.get("deriver"), "commit": zv.get("commit"),
                             "strategy_id": zstrat},
                links=[f"corpus/runs/{z['prev']}"] if z.get("prev") else [],
                body=(f"**Run-zettel {z['id']}** (metaflow {z.get('metaflow_run_id', '?')})\n\n"
                      f"- window: cursor {w.get('harvest_cursor')} · "
                      f"{w.get('passages_fresh', 0)} fresh + {w.get('passages_cached', 0)} cached\n"
                      f"- versions: deriver `{zv.get('deriver')}` @ `{zv.get('commit')}`"
                      + strat_line +
                      f"\n- state: {zst.get('n_chapters')} chapters · "
                      f"EMD {(zst.get('structure') or {}).get('shape_emd')} · "
                      f"congruence {json.dumps(zst.get('congruence'), default=str)[:120]}\n"
                      + (f"- prev: {N.wl('corpus/runs/' + z['prev'], z['prev'])}" if z.get("prev") else "- chain origin"))))
        if zs:
            by_r: dict[str, int] = {}
            for z in zs:
                r = zroots.get(z["id"], "scratch")
                by_r[r] = by_r.get(r, 0) + 1
            print(f"  corpus: {len(zs)} run-zettels (chain head {zs[-1]['id']}; citizenship {by_r})")

    # Trunk lenses (scratch) — same lens ids as the release kasten, root-resolved.
    shape_notes, shapes_line = project_shape_surfaces(live_ids=set(tid2cat))
    notes += shape_notes
    notes += project_lexicon_constructs()
    notes += project_aperture_anchors()
    notes += rel_ch_notes
    notes += rel_coll_notes
    notes += project_escalation_channel()
    notes += project_trunk_lenses(categories, rel_cats, bool(sc),
                                  zettel_head=zs[-1]["id"] if zs else None,
                                  items_report=(assoc or {}).get("report"),
                                  shapes_line=shapes_line)

    for n in notes:
        N.write_note(kb, n)
    entries = (N.scan_notes(kb, "current") + N.scan_notes(kb, "scratch") + N.scan_notes(kb, "archive"))
    idx = N.write_index(kb, entries)
    # content-addressed path manifest (RH 2026-07-21): trails encode against this kasten version
    try:
        from aegir.lineup.path import build_manifest
        pm = build_manifest(kb)
        print(f"  path manifest: version {pm['version']} over {pm['n']} note ids "
              f"(rank-MPH; trails are now durable computation specs)")
    except Exception as _e:  # noqa: BLE001
        print(f"  path manifest: skipped ({_e})")
    by_dp: dict[str, int] = {}
    by_root: dict[str, int] = {}
    edges = 0
    for e in entries:
        by_dp[e["data_product"]] = by_dp.get(e["data_product"], 0) + 1
        by_root[e["root"]] = by_root.get(e["root"], 0) + 1
        edges += len(e["links"])
    print(f"\n  KB projection → {kb}")
    print(f"  {len(entries)} notes  {edges} edges  by_dp={by_dp}  by_root={by_root}  ·  index {idx}")
    print("  lenses: lens/terms · lens/schema · lens/content (per-root — roots are refs)")
    print("  roots: current (the latest release kasten) | scratch (TRUNK — the live projection "
          "+ authored notes) | archive (past releases + snapshots + tombstones)")
    return 0
