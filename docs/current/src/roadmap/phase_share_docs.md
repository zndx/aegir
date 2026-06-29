# Phase — SHARE Docs (browsable corpus in sdg-corpora)

**Status: Phase A DELIVERED (in-aegir mdbook renderer, commit `ae7dbee`, `just kb-mdbook`);
content layer MATERIALIZED (collections in-tree); Phase B (graduate to `sdg-corpora`) PENDING —
the gate is met at Phase B.** · **Decision:** scoped 2026-06-17 · **Depends on:** the lineup
(UI-U0/U1/U2, delivered), the `aegir.lineup sync` SHARE verb (delivered), and a corpus regen
against the current ontology (see [Dependencies](#dependencies)).

This phase turns the public `sdg-corpora` repo from machine-readable artifacts (parquet / TTL /
JSON) into a **human-browsable, cross-linked mdbook document** — the *static SHARE-tier rendering
of the lineup*. Where the aegir gateway lineup is the dynamic, authenticated navigation for us,
this is the portable, public rendering anyone gets by opening the repo on GitHub: no gateway, no
model, no auth.

## Update — collection-structured, chapters-in-tree (2026-06-17, RH)

Two corrections to the original framing, both now in force:

1. **The corpus is the deliverable; it lives in the distribution tree.** The earlier "chapters are
   release/HF assets, not in the repo" was *not* a real decision — `.gitignore` never excluded them;
   they were simply never committed, and a scale-era LFS note got mis-narrated as "deliberately
   leaving deliverables out." The **only** principled withholding is the Atelier **answer-key**
   (`reference.parquet`, the column→SKOS-code scoring key — held out so the blind eval stays valid).
   Everything else — chapters (as text), underlying tables, terms — ships in the tree. (Bulk *binary*
   parquet via LFS/release is a real concern only at ~100K-table scale; not a v0.3 reason to omit
   the actual product.)

2. **The unit is a `collection`, not a flat chapter list.** A **collection** = a FinePDFs-grounded
   **topic** (carried forward from the coverage audit) + its **chapters** (prose + embedded views) +
   the **underlying relational tables** the views project from (semantic-column DDL) + the grounding
   **ontology terms** + a **manifest** of the cross-links. One concept serves as the unit of
   *distribution = navigation = mdbook section*.

**Materialized** (`scripts/build_collections.py`, deterministic): the v0.3 corpus →
`corpus/collections/topic-NNN-<family>/{README.md, chapters/<id>.md, tables/<name>.sql,
manifest.json}` + `INDEX.md`. Current release: **121 populated collections** (1,977 chapters,
4,116 table DDLs) + 79 gap topics (no chapters yet), in-tree. The mdbook renderer (below)
organizes **by collection** (a collection = a book section); Phase A/B otherwise unchanged. Known
caveat this release surfaces: the chapters' *embedded* views are the generation-time (thin) schema,
while the *underlying* tables carry this session's semantic columns — the divergence motivates the
corpus regen.

## The thesis it certifies

> The v0.3 corpus is not just *downloadable* — it is **browsable as a document**. A reader can
> open a chapter, see its rendered view-tables, click through to the relational tables those
> views project from, and from each table click through to the **fully-resolved ontology entry**
> that grounds it — all as static, cross-linked, attribution-clean pages. The same `content ↔
> relational ↔ ontology` lineup primitive (Ward Cunningham's *lineup*, credited; this is a
> navigation primitive, **not** a wiki — no editing/forking) rendered once for us live, and once
> for the public statically.

This is the presentation half of the SHARE boundary: `sync` shares the *data*; this shares the
*navigable document*.

## The artifact (what a reader gets)

An mdbook site (the devenv already ships mdbook + d2/katex/mermaid) with three cross-linked layers:

| Page | Renders | Links out to |
|------|---------|--------------|
| **Chapter** | the chapter prose + its markdown **view-tables**, rendered | each relational table the views project from |
| **Relational table** | the table's columns + types (from `ddl_statements.parquet`), FK edges, a sample | (1) the **ontology entry** it instantiates · (2) FK-linked tables · (3) chapters that view it |
| **Ontology entry** (resolved-template) | see below | the relational table it drives · broader/narrower entries · its family / BFO upper-type |

## Locked design decisions

1. **"Hydrated" = resolved-template** (confirmed). The ontology entry page shows **not** the
   abstract template (`Class: {X:Class} SubClassOf: cco:Artifact, sdg:hasSKU some {Y:Class}`,
   slot placeholders) but the template **resolved for its specific table**: the concrete class
   name, the Manchester axiom with slots named as the actual columns/relationships the table
   instantiates, the SKOS definition (verbalization), the BFO/CCO anchor, the `broader`/`narrower`
   neighbours, and the slot→column mapping that drives the table's DDL. It answers *"what does
   this table mean, ontologically, fully concrete?"* — **not** the corpus row instances (that
   richer instance-hydration is explicitly out of scope for this phase; revisit later if wanted).

2. **The renderer reuses `aegir.lineup.build`; it does not reinvent the cross-link.** `build.py`
   already projects the tri-layer KB (ontology / relational / content notes with `[[wikilinks]]`)
   into `build/dev`. The mdbook emitter (`scripts/render_lineup_mdbook.py`) is an *output target*
   over that projection: KB projection → `book.toml` + `src/SUMMARY.md` + one flat page per note,
   with the lineup `[[id|label]]` wikilinks lowered to mdbook relative links. The SUMMARY leads
   with the collections × lens pivot (the landing), then the collections, the lenses, and the
   ontology / relational / content products — the lineup's panel-trail flattened into a navigable
   document.

3. **Self-contained from published artifacts.** The final renderer reads **only** what ships in
   `corpora/` — `ontology/catalog/*.json` (with `broader` + slot_types + manchester + verbal +
   bfo_anchor), `ddl/<run>/ddl_statements.parquet` (the *already-resolved* tables — so no need to
   port `template_to_table`), and `corpus/`. A reader cloning `sdg-corpora` builds the site with
   **zero aegir dependency**.

## Phasing (A → B) — prove before scaffolding

**Phase A — prove the renderer in aegir. DELIVERED (commit `ae7dbee`, #49).** An mdbook target over
`aegir.lineup`'s projection (`scripts/render_lineup_mdbook.py`, run via `just kb-mdbook` —
optionally `--build` to invoke `mdbook build`) reuses `build.py`'s projection. It outputs the
cross-linked site, lowering wikilinks to relative page links, with the collections × lens landing
at the head of `SUMMARY.md`. Cheap, where the projection + hydration data already live. *No new
devenv, no new repo structure.*

**Phase B — graduate to `sdg-corpora` as a proper project. PENDING (the gate, #50).** Once A is
proven, extract a self-contained renderer into `sdg-corpora`:
- promote `sdg-corpora` to a Python project: its own **devenv**, a **`src/sdg/`** package, and a
  **`just docs-sync`** recipe (mirrors aegir's `just kb-sync` ergonomics);
- the renderer reads only the repo's published artifacts (design decision 3) → emits `docs/` (the
  mdbook book) inside `sdg-corpora`;
- publish the rendered book (GitHub Pages or committed `book/`).

## The gate

**PASS when:** a fresh clone of `sdg-corpora` → `just docs-sync` produces a browsable mdbook site
where every chapter, its rendered view-tables, the relational tables, and the resolved-template
ontology entries are cross-linked and **all links resolve**, built with **zero aegir dependency**.
(Phase A is the in-aegir proof of the renderer + link model — now delivered; the gate is met at
Phase B.)

## Dependencies

- **Corpus consistency.** The docs are only honest once the corpus is regenerated against the
  current ontology — otherwise old-ontology chapter view-tables would link to new-ontology
  entries. **Sequence this phase after (or bundled with) a corpus regen** via the
  `generate_chapter.py` pipeline. Until then, Phase A can run against the current (mismatched)
  artifacts purely to prove the renderer.
- **`sync` already done.** The ontology Data Product is current in `corpora/` as of
  [the SHARE-verb work](../signals_programme.md); this phase consumes those artifacts.

## Scale (deferred)

At v0.3 scale (~hundreds of tables / ontology entries / ~2,000 chapters) a flat mdbook renders
fine. At the production target (~100K relational tables / ~10K chapters) 100K+ static pages need
**sharding / pagination / on-demand rendering** — that design is **deferred to when production
data lands** and must not block the v0.3 browsable release. `log()` any truncation if a cap is
applied at scale.

## Non-goals (this phase)

- Instance-level hydration (corpus rows as ontology individuals) — resolved-template only.
- Editing / forking — the lineup is a navigation primitive, not a wiki.
- Replacing the live gateway lineup — this is its static, public complement, not a substitute.
- Production-scale rendering — see [Scale](#scale-deferred).
