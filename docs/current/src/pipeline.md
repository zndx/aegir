# The Ontology-Grounded Relational Data Generation Pipeline

This chapter documents the project's active track: a closed, membrane-gated
pipeline that turns a window of real-world input text into a
HermiT-certified OWL ontology, its entailed SKOS and SHACL faces, a
relational projection of that OWL, and a dual-register synthetic
textbook corpus — with every artifact provenance-chained back to the
exact input window, strategy, and code that produced it.

One command runs the entire pipeline, idempotent per input window:

```bash
just metaflow    # → src/aegir/flows/sdg_corpora_flow.py
```

The design commitments, stated up front:

- **Content-first, fully derived.** No hand-authored axiom families
  survive. Live axioms are derived from input passages by the engine
  and admitted through membranes, with provenance (pattern / tier /
  `grounds_ddl` / domain / `source_span`). OWL is the source of truth;
  SKOS and SHACL are entailed; SQL is a relational projection.
- **Propose / dispose.** Agents propose; deterministic membranes dispose
  — and every membrane returns its *reason*, so rejection re-prompts
  rather than silently drops. The two strongest membranes (HermiT
  consistency, OntoClean) are un-fakeable.
- **Measured, not asserted.** The output corpus is scored against the
  *input window* over the same retrieval substrate that selected the
  inputs (congruence), and censused against ontology-grounded topics (the
  inverted topic layer) — never against a corpus-fitted model.
- **Idempotent accretion.** Content-hashed documents, a persistent stream
  cursor, per-passage cache keys, and top-up semantics mean re-running the
  flow over the same window is a no-op; advancing the window accretes.

## The nine stages

```d2
direction: right

gen: {
  label: "generate"
  harvest: Harvest\n(FinePDFs window)
  aperture: Aperture\n(ColBERT MaxSim)
  derive: Derive\n(engine + metrology)
  promote: Promote\n(membranes)
  harvest -> aperture -> derive -> promote
}

realizearm: {
  label: "realize"
  realize: Realize\n(HermiT OWL)
  ddl: Relational projection\n+ SHACL shapes
  prose: Prose\n(dual register)
  realize -> ddl -> prose
}

measure: {
  label: "measure + record"
  verify: Verify\n(congruence, gates)
  zettel: Zettel + project\n(lineup)
  verify -> zettel
}

gen.promote -> realizearm.realize: catalog.json
realizearm.prose -> measure.verify: chapters
```

Each stage below names its machinery, its membrane or gate, and the
artifact it leaves behind.

### 1 · Harvest

`scripts/harvest_domain_docs.py` streams FinePDFs through a persistent
cursor; each accepted passage lands as a content-hashed document
(`build/domain_harvest/docs/<sha>.txt`) with an append-only
`manifest.jsonl`. The window is *advanced*, never rebuilt —
`--harvest-target N` streams until N new in-domain documents land, and
content-hash filenames make the operation idempotent by construction.

- **Gate:** only documents classified in-domain by the strategy-declared
  aperture (stage 2) are admitted.
- **Artifact:** the input window — the corpus's outer provenance boundary.

### 2 · The aperture

`src/aegir/ontology/domain_index.py` classifies each candidate document by
qdrant-native ColBERT MaxSim (late interaction) against SKOS domain
concepts — collections `sdg_domains` / `sdg_aperture` — with a
hierarchical argmax gated by `rel_margin`. The aperture is not incidental
plumbing: its collection snapshot (anchor ids, labels, vector hashes) is
the strategy's **lens** pillar, content-addressed so a run can prove which
retrieval surface selected its inputs.

- **Membrane:** the `rel_margin` unambiguity gate — a document that
  cannot commit to a domain does not enter the window.
- **Artifact:** the content-addressed aperture snapshot inside the
  strategy manifest.

### 3 · Derive

The engine (`src/aegir/engine/`, forwarding `instruct` over zndx.engine.v1 gRPC to the federation peer hosting Qwen3.8-27B — workloads
speak only gRPC, never vLLM directly) reads each passage and derives
axiom-pattern-bound primitives: entities with definitions, typed
attributes, and relations, bound to the pattern library
(`src/aegir/ontology/patterns.py`, four tiers: `fhir_minimum` /
`owl2_core` / `odp` / `sysmlv2`). Two harnesses exist over the same
machinery: `scripts/derive_ontology.py` stages catalog candidates into
`catalog.candidate.json`, and the flow's derive step
(`derive_loop.derive_with_metrology`) runs **metrology-informed rounds** —
kvasir profiles the implied DDL against SchemaPile, and a below-`rich`
structural verdict re-prompts the engine with the structural reason,
bounded by `--rounds`.

- **Membrane:** the metrology feedback loop (rich / thin / inert
  verdicts, reasons returned); per-passage stage keys
  (`aegir.strategy.lineage.stage_key`) cache-skip passages already derived
  under the same lens + voices + schema.
- **Artifact:** per-passage entity JSON + Manchester (`entities/`), and
  staged catalog candidates.

### 4 · Promote

`scripts/promote_candidates.py` is the admission boundary into
**`src/aegir/ontology/catalog/catalog.json`** — *the* live catalog
(renamed from `08_derived.json` on 2026-07-07; the hand-authored 01–07
seed families are retired; everything is derived; discovery goes through
`schema.catalog_files()`, never globs). Admission runs a funnel of
membranes: the content membrane
(`src/aegir/ontology/derivation_membrane.py`), the link-1
informed-by-inputs condition, and one aggregate HermiT pass
(consistency + zero unsatisfiable + equivalence dedup) over the batch.

- **Membrane:** the promotion funnel — every drop is printed with its
  reason.
- **Artifact:** the accreted catalog, each template carrying
  pattern / tier / `grounds_ddl` / domain / `source_span` provenance.

### 5 · Realize

`scripts/build_realized_ontology.py` turns templates into concrete OWL
over BFO 2020 + π(CCO) — CCO imported as a *reasoning authority*, so
grounding is validated against its disjointness axioms — and HermiT
certifies the result (consistent, zero unsatisfiable classes). When HermiT
refuses, the boundary emits a *signal*, not a name: minimal unsat
justifications feed the `scripts/reauthor_unsat.py` agent loop
(map → propose → parse+HermiT membranes → apply). Inside the flow,
`scripts/realize_sdg.py` merges the run's entities and emits the
certificate — exit 2 (inconsistent) or exit 3 (unsat) kills the run.

- **Membrane:** HermiT itself, plus the OQuaRE/IOF metrology hard gate on
  the publish path (`aegir.lineup.sync`).
- **Artifact:** `corpora/ontology/sdg-ontology.{omn,owl}` +
  `HERMIT_CERTIFICATE`, and the per-run `ontology/certificate.json`.

### 6 · Relational projection / constructs

`scripts/realize_sdg.py` (and kvasir) lower certified OWL to loadable
SQL and to `shapes.ttl`. Tables, keys, and foreign keys cite class and
property IRIs; SHACL `sh:targetClass` names those same classes.
Per-chapter *constructs* carry the verbatim tables, views, and FKs the
prose must embed. Cross-entity FKs are earned from content, never
name-matched from a family table. `scripts/check_triad_entailment.py`
checks OWL ⊨ SKOS, OWL ⊨ SHACL, and SKOS ⊨ SHACL.

- **Gate:** polyglot validation + the SchemaPile shape-EMD structure
  score (`ontology/structure.json`).
- **Artifact:** `ontology/ddl.sql`, `ontology/shapes.ttl`, per-passage
  construct payloads.

### 7 · Prose

Two register branches run in parallel: **natural** (practitioner domain
prose, tool-free) and **semantic** (ontology-mechanistic prose, equipped
with kvasir tools over MCP and grounded in kvasir-verified facts). Both
are engine-generated with thinking traces retained — the traces are corpus
value, not exhaust.

- **Gate (at join):** the embedded-payload gate — every chapter must
  carry its tables and views (chapters are textbooks-with-embedded-views;
  prose-only is a regression).
- **Artifact:** `chapters/<passage>/{natural,semantic}.md` +
  `.exchange.json` traces, `manifest.jsonl`.

### 8 · Verify / measure

Report-first doctrine: the gates score and record; publishing is a
separate, harder gate (stage 9 → [Data Products](#data-products--the-release-lifecycle)).

- **Congruence** (`src/aegir/ontology/congruence.py`) — the tripartite
  lineage graph: input passages —(harvest MaxSim)→ concept entries
  ←(congruence MaxSim)— output chapters, over the *same* ColBERT
  collection that classified the inputs. This is the BERTopic-era R_D
  reborn on the retrieval substrate: it quantifies how well the corpus
  preserves the input window's concept associations.
- **Sensitive scan** (`scripts/scan_sensitive_nouns.py`) — real
  universals, fictional particulars.
- **Naturalness norms** (`scripts/compare_corpus_naturalness.py`) — the
  re-runnable mechanical-character tracker.
- **Shape EMD** vs SchemaPile — relational realism of the spine.

- **Artifact:** `metrics.json`, `sensitive_scan.json`, the congruence
  report.

### 9 · Zettel + project

Every completed run seals one immutable **run-zettel**
(`src/aegir/lineup/zettel.py`): prev-linked into a chain, citing the run's
`strategy_id`, carrying derive stats and metrics. The lineup then
re-projects (`just kb-build` → `python -m aegir.lineup build`) so the
`/lineup` surface reflects the accretion. Projection is the
failure-tolerant tail — it never loses a corpus.

- **Artifact:** the run-zettel chain + the regenerated KB projection
  under `build/dev/`.

## The measurement instrument — the inverted topic layer

The pipeline's census instrument is the **inverted topic layer**
(`src/aegir/ontology/topic_layer.py`; phase-gated PASS 2026-07-07 — the
[phase gate](./roadmap/phase_gate_inverted_topic_layer.md) is the
authoritative record). The inversion: topics are not fitted to the corpus;
the corpus is measured against the ontology. A topic **≡** a concept
anchor in the qdrant registry (`sdg_topics`: the 29 SKOS domains + all 433
live catalog terms) — the collection *is* the topic registry; there is no
fitted topic-model artifact.

```d2
direction: right

item: Item\n(anchor-proportional\ntoken window)
maxsim: ColBERT MaxSim\nvs sdg_topics
gate: "Hierarchical margin\nrel_margin_h ≥ τ* (0.1065)"
topic: One topic\n(concept anchor)
unassigned: UNASSIGNED\n(error signal)
loop: Definitional-rigor loop\n(membranes M1–M8)

item -> maxsim -> gate
gate -> topic: pass
gate -> unassigned: fail
unassigned -> loop: indicts the lexicon
loop -> maxsim: authored surfaces\n(re-adjudicate, sha-pinned)
```

The load-bearing properties:

- **Items** are token windows sized proportionally to the registry's own
  median anchor length (late interaction is well-conditioned when both
  sides speak at comparable length), char-span-faithful to the source
  document.
- **One item, one topic, or none.** Assignment requires a hierarchical
  unambiguity margin (`rel_margin_h` — margin over the nearest
  *non-ancestor* competitor, so a parent/child near-miss is not
  ambiguity). τ\* = 0.1065 was **pre-registered and derived from a
  shuffled-window null** (report-not-tune), not tuned for rate.
- **The lexicon is the parameter.** Unaligned mass indicts the topics,
  never the input: the engine proposes annotation surfaces
  (`alt_labels`, `scope_note`, definitions), the annotation membrane
  (`src/aegir/ontology/annotation_membrane.py`, gates M1–M6 + M8)
  disposes with a reason, and the M7 basin gate checks the *input side*
  of any registry change. M8 is the durable methodological rule:
  **anchor surfaces are positive-voice only** — definition-by-negation is
  an embedding anti-pattern.
- **Lineage.** Every association record pins the content-addressed
  collection state (`collection_sha`) and lens identity that adjudicated
  it; re-runs against an evolved registry land *beside*, never over,
  prior adjudications. Assigned associations project into the lineup as
  walkable item notes.

This replaces the BERTopic-era instruments (`topic_alignment.py`,
`build_topic_model.py`, `T_I.pkl`), which are retained for v0.3
reproducibility only.

## The strategy and the provenance chain

A corpus artifact is fully determined by three coordinates: the **input
window** (stage 1), the **code** (the commit), and the **strategy** — the
"how it was made" Data Product that captures everything which is neither
input nor code. The strategy lives in the `sdg-strategy` submodule
(`strategy/`), collected by `src/aegir/strategy/manifest.py` in four
pillars:

| Pillar | Contents |
|---|---|
| **lens** | the qdrant aperture — collection snapshots (ids, labels, vector hashes) + the binding that declares runtime targets |
| **voices** | every agent-facing surface: prompts, schemas, feedback templates, tool docs |
| **knobs** | the flow's tunable defaults (HOCON) |
| **targets** | mined norms by content hash + source identity, gate floors |

Identity is Merkle-shaped: `strategy_id = sha256(sorted component
hashes)[:12]`. Shadow strategies are branches (a shadow run gets its own
corpus dir and resolves collections from the ref's binding); promotion is
by cherry-pick. At flow start the live state is drift-checked against the
declared manifest — drift is a warning by default and a hard failure under
`AEGIR_STRATEGY_ENFORCE=1`.

The chain closes at the record layer: *window × strategy × code* → the
run-zettel (immutable, prev-linked, cites `strategy_id`) → the lineup
projection, where every chapter, term, table, and topic-layer association
is walkable back to its origins.

### Both directions of the flow, one adjudicator (2026-07-09)

The instrument reads **both sides**: input passages *and* generated
chapters window and adjudicate through the *same* registry state, encoder,
τ\*, and pinned window — making **input↔output topic-correspondence** a
like-for-like measure for the first time. This supersedes by-construction
template matching, which is *circular* as a measurement (we injected the
templates; finding them verifies injection, not expression). First smoke:
natural-register chapters align at 4.0% vs the inputs' 10.2% at the same
gate — generated prose reads more diffuse per window than its sources, a
generation-quality signal no prior instrument could see. The full build
(both registers, per-topic mass-in/mass-out) is the semantic upgrade of
coverage-close, the binding generator metric.

### Collections as connected relational components (2026-07-09)

The trunk's **collection** unit is being redefined on the founding thesis:
*a collection = the output documents whose tables and views form a
connected DDL graph* (FK + view-composition edges). The raw constructs web
is scale-free — chapter generation performs preferential attachment into
shared lookups — so one hub-glued giant component initially absorbed 58%
of tables. Two network-science filters (Barabási), swept jointly and
validated on the **semantic axis** (within-collection topic entropy from
the item associations): targeted-attack hub removal × weak-ties
(neighborhood-overlap) pruning, with a pre-declared selection rule
(bounded giant component + the partition property: documents touch a
median of **one** component + minimum entropy). Operating point: **160
infrastructure hubs removed** (4.8% of nodes — the scale-free attack
knee; `students`, `persons`, `geographic_regions`… — annotated as shared
infrastructure, referenced by many collections, constitutive of none) →
**94 multi-document collections**, giant component 2.2%, mean topic
entropy **3.17 → 1.14 bits**. The two independent axes — relational
(constructed) and semantic (adjudicated) — certify the unit jointly
(`scripts/relational_collections.py`).

## Data Products & the release lifecycle

The `corpora/` submodule (zndx/sdg-corpora) is SHARE — the published Data
Products: **ontology** (realized OWL + catalog mirror + HermiT
certificate), **ddl** (the released spine), **corpus** (release CARDs),
**vocabulary** (SKOS). Publishing goes through `aegir.lineup.sync` and is
hard-gated (ontology schema CI + the OQuaRE metrology gate) — a flow never
auto-pushes corpora.

**The vocabulary is itself sufficiency-gated (2026-07-09):** the published
`annotations.*` are MaxSim anchor surfaces (Atelier consumes them as a
source taxonomy), so descriptions are **natural** — verbalization frames +
membrane-authored scope notes; the Manchester axiom lives in a reference
column that no anchor composer embeds (433 published descriptions used to
carry raw axiom syntax — machine tokens as attractor mass, the
positive-voice lesson operating at the publishing layer). Every record's
composed anchor text must retrieve **itself** at rank-1 with hierarchical
margin against the whole vocabulary (`scripts/vocabulary_sufficiency.py`
— the self-retrieval gate applied at the know→share boundary; first run:
944/944, median margin 0.20). A definition that cannot discriminate does
not publish silently.

Releases anchor the lineup's three roots (**roots are refs** — RH
2026-07-06). `build/dev/{current,scratch,archive}` is a regenerable KB
*projection* (`src/aegir/lineup/`, `just kb-build`), not a store:

```d2
direction: right

trunk: scratch = TRUNK\nlive catalog · spine · accreting corpus\nzettel chain · strategy · items
current: current = latest RELEASE\ncorpus + era lexicon + DDL + card
archive: archive\npast releases · frozen kastens\ntombstones

trunk -> current: release\n(project the new kasten)
current -> archive: promotion\n(snapshot-freeze)
```

- **scratch** is trunk: the full live projection — live catalog, DDL
  spine, generated web, the accreting corpus, the zettel chain, the
  strategy, and topic-layer items.
- **current** is the latest sdg-corpora release as a complete kasten: the
  released corpus with *its era's* lexicon, released DDL, and card.
- **archive** holds past releases, frozen snapshots, and tombstones
  (retired terms resolve as tombstones — a real trail, never a dead
  link).

Promotion snapshot-freezes current into archive, projects the new release
into current, and trunk rolls on. All three roots expose the same surface
ids (lens/terms twins) with `?root=`-scoped resolution in the gateway; the
UI trail is identical for every root
(see [Lineup Landing](./roadmap/lineup_landing.md)).

## Where the model track consumes this

The pipeline exists in service of the model track
([Introduction](./introduction.md): the byte model is the *target*; this
pipeline is the *active track* producing its substrate):

- **Path A — World-v3 augmentation** (locked; see
  [Pretraining](./pretraining.md)): continue-pretrain a vanilla RWKV-7
  0.19B (`scripts/continue_pretrain_rwkv7.py`,
  `src/aegir/flows/path_a_training_flow.py`) on a World-v3 subsample with
  and without the synthetic corpus, isolating the *data* value of the
  pipeline's output while holding architecture fixed and calibrating the
  safe mixing fraction α.
- **Path B — the Aegir architecture thesis** inherits Path A's calibrated
  α: the H-Net + RWKV byte model trains on the same mix, and the Signals
  programme's final gate compares it against the RWKV-7 baseline and the
  no-ontology ablation ([Signals Programme](./signals_programme.md)).
- **The eventual amortization — latent lenses.** Every topic-layer
  association record is, by construction, a supervision pair for an
  anchor-projection head: the byte model learns to *predict its own
  topic-layer adjudication* (the L2 latent-prediction auxiliary), and the
  lens becomes a fusable specialist state in the swarm substrate. This is
  the pipeline's measurement instrument becoming model capability — see
  [the latent-lens direction](./agent_swarm.md#the-forward-connection-latent-lenses)
  and the [Roadmap](./roadmap.md) next-moves.

The operational entry points, gate suites, and CLI surfaces for everything
above are collected in the [Development Guide](./development.md); the
ontology-side quality machinery is specified in the
[Ontology](./ontology.md) chapter and its
[Authors Guide](./ontology/authors_guide.md).
