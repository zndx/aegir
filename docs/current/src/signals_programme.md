# Signals Programme — Relational Domain Adaptation (aegir workstream)

Status: programme charter (2026-06-16; living — updated 2026-07-09: tri-root lineup, inverted
topic layer, catalog-is-derived). The go-forward development programme for the aegir track,
framed as a workstream of the cross-repo **Signals** initiative. Builds on
[`end_to_end_and_meta_harness.md`](./end_to_end_and_meta_harness.md) (the substrate-evolution machinery)
and the repo-root `EVIDENCE.md` gate discipline. Designed to be lifted into a GitHub project
across the component repos (§Component map).

## Thesis

Signals co-develops a **bounded, verifiable, signal-driven** agent ecosystem (Holland *Signals &
Boundaries*; intrinsic verifiability per Gaius **RASE**). This workstream is its **relational
domain-adaptation engine**, and it rests on three commitments established over the inc-2 work:

1. **The ontology is the domain-adaptive surface** — not weight-bound software. A de-novo-curated,
   reasoner-verified ontology carries domain meaning independent of any model, so it can be evolved
   *before* a trusted model exists (the regime where a meta-harness cannot operate).
2. **The reasoner (HermiT) is the model-independent oracle** — sound & complete, so coherence and
   realization are *ground truth*, not learned proxies. It certifies **formal** correctness; the corpus
   and the model certify **domain** correctness. All three anchors stay live (anti-folie-à-deux).
3. **The H-Net+RWKV model is the ultimate fitness measure** — trained from scratch on the
   ontology-grounded corpus, evaluated on relational understanding and *de novo Data Element elucidation*
   with conventional evals/post-training, deliberately kept as a trustworthy, non-co-adapting arbiter.

The generator (SAE-instrumented **Qwen**, fine-tuned to mint the ontology) and the downstream model
(**H-Net+RWKV**, from scratch) are distinct by design: experiment upstream, keep the instrument boring.

## Source of truth & data flow — the ontology is primary; everything else serves it

The single source of truth is the **generated ontology artifact itself**. Atlas, Qdrant, the tri-root
lineup projection (`build/dev/{current,scratch,archive}`), and even the published `sdg-corpora` are
projections, indices, views, and exports *of* it — never competing stores. The ontology spans two
disclosure tiers: **KNOW** (the full, curated working ontology) ⊇ **SHARE** (`sdg-corpora`, the published
subset). Same artifact, two tiers — not two artifacts.

```d2
direction: down

ontology: "ONTOLOGY — the source of truth\nKNOW (full, curated) ⊇ SHARE (published)"

lineup: "the lineup\nbuild/dev/{current, scratch, archive}\nroots are refs"
atlas: "Atlas\n(synced VIEW)"
qdrant: "Qdrant\nsdg_domains / sdg_aperture / sdg_topics"
corpora: "sdg-corpora\n(the SHARE tier)"

ontology -> lineup: project
ontology -> atlas: glossary-sync
ontology -> qdrant: index
ontology -> corpora: export

atlas -> ontology: edits suggest {
  style.stroke-dash: 5
}
```

- **Dependency arrows point inward.** Regenerating the ontology re-projects the lineup trunk
  (`just kb-build`), re-syncs Atlas, re-indexes Qdrant, re-exports `sdg-corpora`. That inward-pointing
  dependency is what keeps a multi-store assembly coherent instead of a web of drifting masters.
- **The lineup ⟷ Atlas stay in sync *because both project from the ontology*** — not via a direct
  link. There is no lineup↔Atlas channel; both are downstream of the one SoT.
- **Atlas edits *suggest*, they do not commit.** Atlas is a rich glossary-editing surface, but a curator's
  edit there is a **proposal** that round-trips into the ontology's curation queue (reviewed, reasoner-gated,
  applied), then re-projects outward. So the Atlas glossary-sync is a *suggestion-returning projection*, not
  an authoritative store — its write path is a PR against the ontology, never a commit to it. (Same shape as
  trunk → release promotion; Atlas is just another suggestion inbox alongside the generator's minted
  candidates and the authored scratch notes.)
- **The lineup is the navigate / explore / curate layer** — the one place ontology-projection, Atlas-sync,
  and the Qdrant retrieval text are seen together, and from which curation decisions are made. Its three
  roots are **refs** (RH 2026-07-06): `current` = the latest sdg-corpora **release** as a complete kasten
  (released corpus + its era lexicon + released DDL + card); `scratch` = **trunk** (the full live
  projection — live catalog, DDL spine, generated web, accreting corpus, zettel chain, strategy, items);
  `archive` = past releases + frozen snapshots. Promotion = snapshot-freeze `current` → `archive`, project
  the new release, trunk rolls on; the same surface ids resolve per root (gateway `?root=`).

**The verification membrane (corollary of commitment #2).** The anchors partition verification by *where
reality is authored*. **Inside** the loop — the ontology and everything projected from it (DDL spine, corpus
structure) — verification is **intrinsic**: HermiT is the oracle, correctness is *proved*, not checked.
Validation-after-the-fact (shape checkers, assertion contracts, external catalogs) *compensates* for not
generating-correct-by-construction and has no place inside the boundary. Extrinsic verification is legitimate
**only at the membrane** where un-generated reality enters — grounding the corpus against FinePDFs (R1) and
the held-out downstream eval — where the system honestly tests itself against a world it did not author. So
anchors #2 (formal / intrinsic) and #3 (domain / membrane) are both required and *not* redundant: they verify
opposite sides of the boundary. **Consequence — integrate freely, depend on nothing.** Under in-situ RASE
engineering, switching costs across the metadata/governance plane (OpenMetadata, OpenLineage, Atlas, Ranger,
…) are ~zero, so integration carries no lock-in — which *licenses* deep, enthusiastic integration rather than
abstention. We implement the standing directives in full: **OL/Marquez compatibility** (external tooling sees
our lineage events) and **Atlas deep integration** (a richly extended glossary/lineage surface). This is
*non-dependence, not non-integration*. The discipline that keeps it non-dependent: the ontology is the single
source of truth, so Atlas and every projection stay *rebuildable-from-the-ontology-or-it's-a-bug, never a
master* (Atlas edits suggest, never commit). We sample patterns and adapt protocols on our terms; we owe the
plane no gravity — and that freedom is exactly what makes integrating with it generously *safe*.

**The maturation arc — substantially landed.** The seed-crystal stage is over: the hand-authored seed
families are retired and the catalog is fully **derived** (`src/aegir/ontology/catalog/catalog.json`,
accreted by the membrane-gated derive → promote loop; discovery via `schema.catalog_files()`). The
terms-are-real end state arrived as the **inverted topic layer** (2026-07-07, phase-gated PASS —
[the gate record](./roadmap/phase_gate_inverted_topic_layer.md) is authoritative): topics **≡**
ontology-grounded concept anchors in Qdrant (`sdg_topics`: the 29 SKOS domains + all 433 live catalog
terms; `src/aegir/ontology/topic_layer.py`), items = anchor-proportional passage windows, one item → one
topic via hierarchical-margin MaxSim or UNASSIGNED — and the **lexicon is the parameter**: unaligned
input indicts the topics, never the input (the inverted-LDA direction). Every association is
content-addressed to the collection state that adjudicated it, so accumulated lineage survives registry
evolution. The corpus-fitted BERTopic-era topic model (`topic_alignment.py`, `build_topic_model.py`,
`T_I.pkl`) is **deprecated** — v0.3 reproducibility only; R_D's successor is congruence
(`src/aegir/ontology/congruence.py`) + the topic layer. The end state stands and is now measurable:
**the lineup ≡ Atlas glossary ≡ Qdrant index — three views of one real ontology.**

**The two faces that make a term "real":** (1) **Qdrant augmentation, recorded as SKOS annotation
properties** *(built)* — per the BERTSubs §4.3.2 multi-label technique, each term's distinguishing
text-features live *in the ontology* as `skos:prefLabel` / `skos:altLabel` (the BERTSubs multi-label set /
MaxSim match surfaces) / `skos:definition` / `skos:scopeNote` / `skos:example` — common SKOS constructs with
domain values, not novel ones. The lineup panel renders them, they assemble into the ColBERT/MaxSim
retrieval text, and `L(c1)×L(c2)` over the altLabel sets multiplies the BERTSubs subsumption pairs (→ the
hierarchy edges for per-term-panel navigation). The surfaces are now **authored through a closed membrane
loop** (`scripts/author_skos_surfaces.py`): the engine proposes `{alt_labels, scope_note, definition}` per
term; the annotation membrane (`src/aegir/ontology/annotation_membrane.py`, gates M1–M6 + **M8: positive
voice only** — definition-by-negation is an embedding anti-pattern) disposes with a reason; the result is
verified by the self-retrieval margin gate plus the input-side M7 basin gate — the annotation-layer oracle,
distinct from HermiT on the axiom layer (annotations never assert taxonomy). The **subsumption hierarchy** over
those terms is now realized *autonomously* by `mediate_hierarchy` (`scripts/`, built 2026-06-17): mpnet
candidates → Grok proposes (ACP) → a two-layer gate — HermiT consistency/coherence/acyclicity **and** domain
vocabulary overlap — admits only verified edges, **no human review** (the tools are the arbiter). This is a
RASE-pattern increment: the agent realizes the hierarchy capability through the meta-harness, intrinsically
verified. The lineup renders the result as per-term **Broader/Narrower** navigation. (2) **Atlas glossary-sync** — the
`AtlasGlossaryTerm`/`Category` projector (extending the existing `rdbms_*` relational projector), a
*suggestion-returning* projection keyed on `qualifiedName`, carrying the same SKOS annotations as term
attributes.

## Methodology — factored gates over a scaling ladder

The programme is a **high-dimensional, multi-objective** optimization decomposed into a **sequence of
low-DOF gates**, valid because the factorization respects the problem's interaction structure:

- **1-DOF where separable** (architecture ⊥ data-mix; scale is a ladder, not a competing knob).
- **2-factor where the interaction is the hypothesis** — specifically α (ontology-corpus fraction) × β
  (SQL/DDL fraction): semantics × syntax of relations, the most likely super-additive effect.
- **Report Pareto slices, don't scalarize** — each gate's output is a frontier (general vs relational vs
  DE-elucidation); the operating point is chosen once the surface is mapped, not at gate 1.
- **No scaled spend without a green gate** (the standing EVIDENCE rule). Each milestone is a
  pre-registered EVIDENCE entry; a failure localizes to a *named arrow*, not a diffuse "it didn't work."

## Component map (the eventual GitHub project)

| Repo | Role in this workstream |
|------|-------------------------|
| **aegir** | the engine — ontology + reasoner + meta-harness + H-Net+RWKV + pretraining/eval |
| **gaius** | RASE metamodel + MetaAgent — the shared verifier discipline + calibration precedent |
| **corpora** (`sdg-corpora`) | the corpus / SKOS / DDL artifact — the publishable deliverable |
| **oss-polyglot** | the SQL/DDL syntactic axis — the β data amendments |
| **atelier** | independent pre-training efficacy gate (blind classification, reference withheld) |
| **asf-atlas** | provenance / lineage — the digital thread across the pipeline |
| **hnet**, **rwkv-lm** | reference architectures (dynamic chunking; RWKV-7 baselines + open corpus) |
| **cldr/signals** | the umbrella — boundary/signal contracts; the GitHub project's home |

## Milestones

**Status (2026-07-09).** M0 remains the last closed milestone; **M1 is the gated next spend** (GPU), with
Path A (World-v3 augmentation: continue-pretrain vanilla RWKV-7 0.19B ± the grounded mix) locked as the
data-value harness whose calibrated α Path B inherits. Everything landed since M0 sits on the instrument
side, under the no-scaled-spend rule: the corpus pipeline became one idempotent flow (`just metaflow` →
`src/aegir/flows/sdg_corpora_flow.py`; [the pipeline chapter](./pipeline.md)); Track A relational realism shipped with the token-yield bound
measured early (Y_eff ≈ 7.9M — M2's "bound to quantify"); the refinement loop closed, scaled, and staged
the v0.4 corpus ([refinement loop](./refinement_loop.md)); the realized ontology publishes under the
OQuaRE/IOF hard gate; and the **inverted topic layer** passed its phase gate — R1's topic instrument is
live and BERTopic-free ([gate record](./roadmap/phase_gate_inverted_topic_layer.md)).

**M0 — Foundation (DONE, 2026-06-16).** The substrate-evolution machinery: HermiT coherence gate
(inc-2a), single-file harness (inc-2b), Meta-Harness outer loop + first discovered harness (inc-2c),
realization-as-CPA beachhead (inc-2d). Committed; see EVIDENCE.md. *The reasoner gates and computes; the
harness evolves.*

**M1 — Architecture baseline (the H-Net isolation gate).** Train H-Net+RWKV on RWKV-7's open corpus,
swapping **only** the tokenizer for byte-level dynamic chunking; establish parity across the scaling
ladder to RWKV-7-matched params. **DOF = 1** (the chunking change), corpus held constant.
*Gate: H-Net+RWKV ≥ RWKV-7 at matched scale on standard evals → architecture certified, isolated.*

**M2 — Instrument validity (the decisive corpus gate + proxy calibration).** A same-architecture,
matched-budget matrix at ≥2 ladder rungs:
- **arms:** grounded mix / no-ontology ablation / standard-only (the ablation arm already exists).
- **α×β 2-factor cell** replicated at two scales — one design yields both the interaction sign *and* the
  scale-drift of the mix optimum.
- **eval:** FLOOR (grounded ≈ standard on general LM evals → **non-degeneracy**, the failure mode every
  upstream proxy is blind to) + LIFT (grounded > ablation on relational + DE-elucidation; cells-only,
  control tasks, PR-metrics, bootstrap CI).
- **side-product:** calibrate the cheap proxies (R1 / coverage-close / corpus-quality) against the
  pretrain signal — the E1 / RASE calibration loop, run once to certify the proxies that drive iteration.
- **bound to quantify:** the corpus's max non-repetitive token yield (caps α at scale).
*Gate: floor held AND lift CI-clean AND α×β interaction + scale-drift characterized.*

**M3 — Scale + the Final Phase Gate.** Conditioned on M2 green: climb to RWKV-7-matched params,
extrapolate α*(N)/β*(N) to target scale, confirm the lift persists. (Gate text below.)

**M4 — The forward door (unlocked by the final gate).** Iterate depth/breadth of the machinery:
(a) close the generator loop — SAE-instrumented Qwen with **process-reward** fine-tuning (anti-Goodhart:
reward the ontological-reasoning circuit, not just the verdict), the mutually-affirming ontology↔generator
cycle anchored by the downstream model; (b) **RASE in a novel domain** — apply the pipeline to a second
information domain with *minimal re-tuning* and measure what breaks (the topic layer/lexicon, the
aperture, BFO anchoring, congruence/R1). Promotes "valid instrument" → "validated method."

## Final phase gate

> **At RWKV-7-matched scale, the ontology-grounded data mix yields an H-Net+RWKV model that**
> **(a) matches RWKV-7 on general/standard evals — the non-degeneracy floor — AND**
> **(b) exceeds the no-ontology-ablation control on relational understanding + de novo Data Element**
> **elucidation, CI-clean, with the α×β interaction and the mix-optimum scale-drift characterized.**

Passing **certifies the ontology machinery as a valid instrument for relational domain adaptation** and
authorizes the RASE-generalization phase (M4 → novel domains). Failing localizes the break to a named
arrow — ontology→corpus degeneracy (M2 floor), corpus→model transfer (M2 lift), or scale-drift (M3) —
each of which has its own remediation. This gate supersedes the proxy-only corpus-as-deliverable gate: the
proxies are *calibrated by it*, not trusted ahead of it.

## Three external anchors (held live throughout)

The symbolic co-evolution (ontology ↔ generator ↔ harness) optimizes formal + proxy signals only; three
non-co-adapting anchors keep it honest: the **reasoner** (formal ground truth), the **corpus** (empirical
fit), and the **held-out H-Net+RWKV** (behavioral domain truth). No milestone closes on proxies alone.
