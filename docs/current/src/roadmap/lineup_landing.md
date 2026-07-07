# Lineup Landing — the collections × lens pivot

**Status: BUILT (the spec that shipped).** · Co-designed with RH 2026-06-17/18; landed as the lineup's
`current` beachhead (collections × lens pivot + the de-flattened many-to-many graph + the TF-IDF lens
chords — see §Implications). Defines what a user sees on clicking **lineup**, and the data model the view
sits on. Distinct from (but shares its graph with) the [SHARE-Docs phase](./phase_share_docs.md).

## What the user is seeking — a trailhead, not a dashboard

Cunningham's lineup is *trail-following* (it is a navigation primitive, **not** a wiki — no
edit/fork). So clicking "lineup" almost never means "show me a report" — it means **"give me a good
place to start a trail."** The landing's job is to offer the strongest *trailheads* and get out of
the way. **Presume: orientation → first trail step.**

```gherkin
Scenario: Orient (the default, first-visit intent)
  Given I open the lineup with no specific target
  When the `current` root loads
  Then I see collections as rows under the default lens, with the shape of the KB
  So I can pick a direction in one glance.

Scenario: Browse the corpus (the richest axis)
  Given I want to explore content
  When I read the landing
  Then collections are first-class rows, each a topic-grounded bundle
  So I can dive into one (its chapters, tables, terms).

Scenario: Re-orient by lens
  Given I am looking at collections under the `terms` lens
  When I switch the lens to `schema` (or `content`)
  Then the column axis swaps (terms → base-tables → topics) while collections stay the rows
  So I see the same collections through a different facet without losing my place.

Scenario: Transpose
  Given a cell links a collection to a term (or topic, or table)
  When I follow it from the terminal side
  Then I get "this term → the collections that realize it" (the inverse trail)
  So the pivot reads both ways.

Scenario: Seek a known thing
  Given I know what I'm after
  When the landing loads
  Then a jump/search box is focusable immediately
  So I skip traversal.

Scenario: Resume
  Given I was mid-investigation
  When I open `scratch` from the root dropdown
  Then I see my authored notes / recent trail
  So I continue where I left off.
```

## The beachhead: collections × lens

`current` = a pivot. **Collections are the fixed rows**; the **lens selects the column dimension**
(`terms` default → `schema` → `content`). **Cells are incidence — i.e. trailheads** (optionally
carrying a count). The lens swaps *what the columns are*; the unit never changes. This is the user's
"compound / pivot-table" intuition, made precise: the lens is the *column-axis selector*, not a
co-equal second data axis.

## The truth underneath: a multi-hop graph, every terminal many-to-many

The pivot is a **surface** over a graph whose hub is the **document** and whose relational join-node
is the **view**:

```
collection ──member──▶ document ──realizes/cites──▶ term
                               ──target+style──────▶ topic
                               ──embeds──▶ view ──hydrates from──▶ base table
```

Each lens reaches a *terminal* (term / topic / base-table) **through** an intermediary. Flattening
any intermediary collapses a real many-to-many into a false diagonal or many-to-few — which is
exactly what an early materialization (`build_collections.py`, grouping on the single
`target_topic_id`, going straight to base tables) did:

| Lens | Intermediary (must not be flattened) | Terminal | Cardinality vs collections |
|------|--------------------------------------|----------|----------------------------|
| **terms** | document (realizes) | term | many-to-many |
| **content** | document (`target` **+ `style_topic_ids`**) | topic | many-to-many |
| **schema** | document → **view** | base table | many-to-many |

**The data already proves it** (v0.3 corpus, before any de-flatten):

- **Topics:** 160 distinct topics touched; **160 / 160 already span > 1 collection**; one spans **32**;
  a chapter touches ~3 topics (target + style).
- **Base tables:** 146 cited; **95 (65%) already feed > 1 collection**; one feeds **35**; avg 4.5.

So there is **no diagonal and no many-to-few** — those were dropped-hop artifacts. Every terminal is
many-to-many because the **document is a shared hub** and the **terminals recur** (a topic styles many
documents; a base table hydrates many views). The "collection ≈ topic" 1:1 in the current release is a
**degenerate, transitional instance** of this graph, not its shape.

> The rule, found independently on two axes: **the view is to the schema axis what `style_topic_ids`
> is to the content axis** — the intermediary that, surfaced, turns a flattened projection back into
> the real graph.

## First-class entities

The model requires two entities that are currently *implicit* to become first-class lineup nodes:

- **document** — the universal hub (everything hangs off it: terms, topics, views).
- **view** — the relational join-node (`view → base tables`; a view may source tables across
  topics/PDFs; a base table feeds many views). This is the project's founding thesis made navigable:
  *a corpus table is a view on a larger, shared relational footprint.* The Atlas projection already
  models views (`view_<table>`, `join_<a>__<b>`); the lineup/collections layer must too.

Plus the terminals (term, topic, base-table) and the unit (collection).

## Lens renderings (all pivots; the card is orthogonal)

- **terms** (default — leads with the ontology grounding, the thesis): `collection → terms`, invertible
  to `term → collections`.
- **schema**: `collection → documents → views → base tables` — the shared footprint; **pivot center is
  the view**. Surfaces both per-document views and the shared base tables.
- **content**: `collection ↔ topics` (the densest cross-axis). The **README-style "expanded topic"
  card is the *drill-in detail* of one collection — not the lens itself.** Pivot → click a cell →
  card. Two layers; don't collapse them (that collapse was the 1:1 artifact).

Affordances the pivot grants for free: **transpose** (read either direction), **counts as density**
(orientation before drill-in), **cell = trailhead** (keeps it a lineup, not a report).

## Tri-root: a layer dropdown, not a browse axis

Pull `archive / current / scratch` into a compact **top dropdown, `current` default** (plain
labels — Archive / Current / Scratch). It selects the *layer*, not the browse axis, so it
shouldn't eat left-nav space.

**Root semantics (RH 2026-07-06 — supersedes both the original "snapshots + aged" reading and
the interim "unreleased-chain-only scratch"): the roots are REFS, git-style. Each root is a
kasten with the SAME surface shapes (`lens/terms`, `lens/schema`, `lens/content`, corpus
surfaces); note resolution is root-scoped (`/api/kb/note/<id>?root=` prefers the active root's
instance, falls back cross-root so trails always resolve).**

- `current` — the **latest sdg-corpora release**, navigable as a complete kasten: the released
  corpus (chapters · topics · collections), the release-era lexicon (the catalog generation
  that corpus cites), the released DDL spine (`corpora/ddl`), and the release card.
- `scratch` — **trunk**: the full live projection, where incremental advancements land as
  they happen — the derived catalog and its deterministic spine, the accreting `just metaflow`
  corpus (constructs web + run-zettel chain + strategy), plus hand-authored working notes
  (`scratch/<iso-date>/…`). Work toward the *next* release proceeds here, permanently.
- `archive` — **past releases** (their cards; full kastens once frozen), quarterly
  `kb-snapshot` zettelkastens, aged authored notes, and tombstones (recovered catalog ids
  cited by nothing current).

**Promotion runbook** (when the next release is cut): freeze `current/` into a namespaced
archive kasten (`just kb-snapshot --key <release>` — the machinery below), project the new
release into `current`, and leave `scratch` untouched — trunk keeps rolling. The sdg-corpora
dataset CARDs (`corpora/corpus/CARD_v*.md`) are the release records the projector reads;
`zettel.chain_roots` stands ready to compute run-zettel citizenship from `kind: "release"`
cut points in the chain.

The same sidebar layout (LENS / TRAINING groups — no corpus group; corpus surfaces are
reached through the content/schema lenses) serves every root. TRAINING appears on `scratch`
too: the procedures run against the in-progress corpora as trunk iterates toward the next
release (the training/metrics notes are twinned current+scratch). In `archive`, the groups
browse the **newest frozen kasten** (each snapshot was "the lineup when it was current", so
it carries the same surfaces under its key prefix — full navigation parity with `current`).
Older freezes are retained on disk and stay reachable by id, just not via the side-nav.
Kastens are discovered by their `<key>/lens/terms` surfaces, not registry notes (the registry
pointer notes were removed — RH 2026-07-06); live embeds (lens chords, training viz) are
suppressed inside frozen kastens.

## Loading

Render the **trailhead structure immediately** (lenses + collections + search are known from the
index at once); stream **counts** and **recents**. Orientation starts before the data fully lands —
the skeleton is the trailheads, not spinners.

## Implications & dependencies

The three implications below are **DONE** in the KB projector (`src/aegir/lineup/build.py`): the lineup
projection — not the standalone `scripts/build_collections.py` release materializer — is now the live
source of the landing's graph.

1. ✅ **De-flatten the materializer.** `build.py::project_collections` carries the intermediaries the
   early `build_collections.py` materialization dropped: `style_topic_ids` (→ content many-to-many) and the
   **view** layer (→ schema many-to-many). One change of *kind* ("stop collapsing the hub and the view"),
   not three axis-specific ones (tasks #52/#53).
2. ✅ **Model `document` and `view` as first-class lineup notes** (with their edges) so the pivot is
   computed over the true graph; `project_lenses` emits the collections × {terms, schema, content} pivots
   over it, and the TF-IDF lens chords (`aegir.viz.lineup_app`, task #54) render the live associations.
3. **Shares the graph with [SHARE-Docs](./phase_share_docs.md)** — the mdbook renders the same
   collection ↔ document ↔ {terms, topics, views→tables} structure; both are built against this model.

## Versioning — namespaced archive snapshots

Before a regen (or any version cut), `just kb-snapshot --key <key>` freezes `current/` into a
**namespaced, self-contained** zettelkasten under `archive/<key>/`: every note id and `[[wikilink]]`
is prefixed `<key>/`, so the snapshot coexists with the regenerated `current` (no id collision) and is
internally navigable (clicking inside stays inside). The Archive sidebar's SNAPSHOT selector
discovers a kasten by its `<key>/lens/terms` surface (registry notes are no longer used for
navigation); a `_manifest.json` pins the corpus/coverage/catalog it was projected from
(reproducible). The snapshot survives `kb-build` (which rebuilds `current` only).

**2026Q2 — pre-regen snapshot** (taken 2026-06-18, calendar quarter): 3,397 notes · corpus
`sdg_corpus_v0_3/d7646714…` · catalog `ae7dbee`. Regenerate with
`AEGIR_CORPUS_RUN=…/d7646714…/chapters.parquet just kb-build` at catalog `ae7dbee`, then
`just kb-snapshot --key 2026Q2`. This preserves the pre-regen lineup so the corpus regen is safe.

*Scale note:* merging a full snapshot into the live index ~doubles it per snapshot. Fine for one;
when snapshots accumulate, move to per-snapshot frozen indices mounted on demand (keep the live index
`current`-only) — the deferred refinement.

## Non-goals

- A wiki (no edit/fork — it's Cunningham's lineup).
- Replacing the live gateway lineup — this *is* the live lineup's landing.
- A numeric-aggregation pivot — cells are incidence/trailheads (counts are an optional density hint).
