# Roadmap

The project converges **two coupled artifacts** that share substrate and
are cited together: a **byte-level sequence model** for relational
metadata understanding, and a **BFO/CCO-grounded ontology + synthetic
corpus** that grounds and supplies its pretraining data. They are
coupled by design — the ontology is the *domain-adaptive surface*, the
H-Net+RWKV model is the *ultimate fitness measure*, and the reasoner
(HermiT) is the model-independent oracle between them. A better ontology
yields a better corpus yields a better model; the model's behaviour, in
turn, is the gate that certifies the ontology machinery.

The current go-forward programme — the milestone ladder, its gates, and
the scaling discipline — is the
[**Signals Programme**](./signals_programme.md), of which this page is the
landing view. The end-to-end pipeline is documented stage-by-stage in
[The Relational Data Generation Pipeline](./pipeline.md); the
agent-mediated machinery that runs it is specified in
[End-to-end + Meta-Harness + reasoner](./end_to_end_and_meta_harness.md).
This page summarises what is **delivered**, what is **in flight**, and
what is **gated**, and points to the authoritative reference for each.

> Two earlier framings have been **reorganized under the Signals
> Programme — not retired.** (1) The four-phase K2.5 PARL roadmap
> (supervised → reward → PARL → swarm RL) folded into the Signals
> milestone ladder; the swarm modules persist as infrastructure and the
> RL approach evolves through the GEPA / Agent-Lightning layers (see
> [Agent Swarm](./agent_swarm.md)). (2) The *RLVR-for-ontology-generation*
> line is **long-horizon, not superseded**: its four-component verifier
> *R(O, I)* is now **realized as the deterministic membrane stack**
> (HermiT/CCO, OntoClean, OQuaRE) built this cycle, and an
> SAE-instrumented-Qwen generator fine-tuned by GRPO against that reward
> is the **Signals M4** apparatus for autonomous, local ontology
> extension. The agent-mediated **propose/dispose** loop is building and
> proving that reward *now* — current work in direct service of M4. The
> [Concept brief](./ontology/concept_brief.md),
> [Semantic engine reference](./ontology/production_state.md), and
> [RLVR chapter](./ontology/rlvr.md) carry the M4 research design; the
> reward-modeling / PARL / swarm-RL design notes remain in the tree.

## Delivered substrate

The model track produced a working, air-gap-deployable training and
evaluation stack, and a first real backbone.

- **End-to-end training pipeline.** BDD-backed training on the
  `gt-signals-dbpedia` benchmark (120 DBpedia labels, 814 tables), with
  per-epoch boundary diagnostics and checkpoint discipline
  (`outputs/runs/{run_id}/…` with sidecars and pre-rendered Bokeh plots).
- **Gateway + leaderboard.** A FastAPI gateway (port 8091) + React UI +
  the ABI-patched flash-attn / mamba-ssm build path. The leaderboard
  reads `outputs/runs/` directly — no tracking daemon, no W&B, no
  MLflow. Deployment targets: devenv (local), CAI (PGlite), Zarf
  air-gap K8s.
- **v2 mixed-corpus pretrain (2026-04-27).** The project's first real
  byte-level backbone — 122k training steps on a mixed corpus
  (FineWeb-Edu + SQaLe + SchemaPile + FinePDFs-lab), single GPU.
  Stratified held-out evaluation shows non-degenerate representations
  across the trained-time slices, with bits-per-byte drops on
  domain-targeted data and general prose held flat. See
  [Training Regime §10](./training_regime.md) for the full table.
- **Synthetic-corpus byte pretraining imparts transferable column
  skill.** On a cells-only GitTables-DBpedia CTA frozen-backbone probe,
  a corpus-pretrained backbone reaches 0.66–0.70 accuracy vs. ≈0.12 at
  random init, with non-overlapping bootstrap CIs and rising
  Hewitt-Liang selectivity. *Caveat:* per-column CTA did **not**
  separate the full / no-ontology / no-schema ablation arms — flat CTA
  is surface-solvable — so the ontology's load-bearing claim moved to
  the relational axis (the M2 lift, below). See the repo-root
  `EVIDENCE.md` (E1–E3, "already-supported claims").

These artifacts are the read surface and the warm-start for everything
below.

## The ontology + corpus, and its rigor program

The ontology is the **annotation vocabulary for Column Type / Column
Property Annotation (CTA/CPA)** over wide relational tables. It is
**content-derived from FinePDFs** (the qdrant/ColBERT aperture,
`src/aegir/ontology/domain_index.py`), then realized to a
**HermiT-validated OWL artifact** at
`corpora/ontology/sdg-ontology.{omn,owl}` (with `HERMIT_CERTIFICATE.md`).
Its classes are **intermediate-depth subsumers** — the property-bearing
classes a heterogeneous-but-coherent column belongs to — not leaf terms.

How it is authored, and every metric, gate, and membrane that governs
it, is the subject of the canonical
[**Ontology Authors Guide**](./ontology/authors_guide.md). The standing
discipline is **propose / dispose**: an agent (or a human) *proposes*
axioms; a stack of deterministic membranes *disposes* — a **parse**
membrane (OWLAPI well-formedness), a **reasoning-authority** membrane
(CCO imported, HermiT validates grounding against CCO's disjointness
axioms in the loop), and an **OntoClean** meta-property membrane
(anti-rigid-cannot-subsume-rigid). The two strongest are *un-fakeable*;
a failure returns its reason, and the author refines (the
agent-mediated feedback loop).

**The rigor program (delivered, gate GREEN).** Benchmarked against the
IOF/BFO signature, the realized ontology is metered by
`scripts/ontology_metrology.py` (IOF rigor dimensions —
`definitional_completeness`, `bfo_grounded`, `realizable_machinery`,
`def_annotation_coverage`; OntoQA/OQuaRE structural metrics; OntoClean
taxonomic-correctness proxies) and gated by the **OQuaRE quality gate**
(`scripts/ontology_oquare.py` — six SQuaRE characteristics, IOF-anchored
[1,5] bands, floors `aggregate ≥ 3.5` and `FunctionalAdequacy ≥ 3.0`
plus HermiT-consistent, aim 3.9, wired HARD into
`aegir.lineup.sync._gate`). Two pre-registered objectives,
**OQ-Structure** and **OQ-Rigor**, are both **MET**: current
`definitional_completeness` 0.554, `bfo_grounded` 0.896,
`realizable_machinery` 10, **0 unsatisfiable classes**, **OQuaRE 4.24 —
GREEN**. The standing rule: no `sync --push` of the ontology Data
Product below GREEN. See the repo-root `EVIDENCE.md` (OQ-Rigor /
OQ-Structure).

**The corpus pipeline + DDL spine (delivered; phase gate PASSED
2026-06-07).** `just metaflow` is the entire corpus pipeline,
idempotent per input window (`src/aegir/flows/sdg_corpora_flow.py`):
harvest (FinePDFs, cursor-windowed) → the aperture (qdrant ColBERT
MaxSim, `src/aegir/ontology/domain_index.py`) → derive (the gRPC
engine over the axiom-pattern library, `src/aegir/ontology/patterns.py`)
→ membrane-gated promotion into the live catalog
(`src/aegir/ontology/catalog/catalog.json`) → realization
(HermiT-certified OWL) → the **DDL spine** — referential-integrity-true
tables / views / FKs, polyglot-validated (Trino ∩ Spark), projected
into Atlas ([Phase Gate — Governance & DDL
Spine](./roadmap/phase_gate_governance_ddl.md)) → dual-register prose
chapters with retained thinking traces → verification and measurement
(congruence — input-window concepts ↔ chapter concepts over the same
ColBERT substrate, `src/aegir/ontology/congruence.py` — plus the
sensitive scan, naturalness norms, and shape EMD vs SchemaPile) → one
immutable, prev-linked run-zettel per completed run
(`src/aegir/lineup/zettel.py`, citing the run's `strategy_id`) and a
lineup re-projection. The corpus is the byte-pretraining data and an
independent publishable deliverable (`corpora/`, the `sdg-corpora`
submodule). **Path A** (`src/aegir/flows/path_a_training_flow.py`) is
the continued-pretraining augmentation on RWKV World v3 — the locked
data-value harness whose calibrated α Path B (the Aegir architecture
thesis) inherits.

**The inverted topic layer (R1 — delivered; phase gate PASSED
2026-07-07).** The live topic instrument
(`src/aegir/ontology/topic_layer.py`): topics **≡** ontology-grounded
concept anchors in qdrant — 462 anchors (the 29 SKOS domains + all 433
live catalog terms), content-addressed by collection sha — and items
are anchor-proportional passage windows that align to **one topic or
none** via hierarchical-margin MaxSim under the pre-registered,
null-calibrated gate (τ\* = 0.1065, the shuffled-window-null p95,
report-not-tune). Ambiguous mass is the error signal and the
**lexicon is the parameter**: anchor surfaces iterate through
membranes M1–M8 that return their reasons (M8 — positive voice only —
is the durable methodological rule). Every association is pinned to
the collection state that adjudicated it and walkable in the lineup.
This is R1's topic layer: the corpus census over ontology-grounded
topics re-runs per input window with no topic-model re-fit; the
BERTopic-era instruments (`topic_alignment.py`, `build_topic_model.py`,
`T_I.pkl`) are deprecated to v0.3 reproducibility. Full evidence table
and methods:
[Phase Gate — The Inverted Topic Layer](./roadmap/phase_gate_inverted_topic_layer.md).

## Go-forward — the Signals Programme milestone ladder

The programme is a high-dimensional, multi-objective optimization
decomposed into a sequence of **factored, low-DOF gates** over a scaling
ladder; the standing EVIDENCE rule is **no scaled spend without a green
gate**. Full charter, the α×β interaction design, and the final-gate
text are on the [Signals Programme page](./signals_programme.md); the
pre-registered hypotheses, instruments, and decision rules are in
`EVIDENCE.md` (repo root).

- **M0 — substrate-evolution machinery. DELIVERED (SUPPORTED).** The
  reasoner gates and computes; the harness evolves: the HermiT
  consistency gate (inc-2a), the single-file harness H₀ (inc-2b), the
  Meta-Harness outer loop + a discovered harness H₁ (inc-2c), and the
  realization-as-CPA beachhead (inc-2d). See
  [End-to-end + Meta-Harness](./end_to_end_and_meta_harness.md) and the
  §Meta-harness entries in EVIDENCE.md.
- **M1 — architecture baseline (the H-Net isolation gate). UNTESTED —
  the next gated milestone.** Train H-Net+RWKV on RWKV-7's open corpus,
  swapping **only** the tokenizer for byte-level dynamic chunking, and
  establish parity up the scaling ladder to RWKV-7-matched params.
  **DOF = 1.** *Gate:* H-Net+RWKV ≥ RWKV-7 at matched scale on standard
  evals.
- **M2 — instrument validity (the decisive corpus gate + proxy
  calibration). UNTESTED.** A same-architecture, matched-budget matrix at
  ≥2 ladder rungs — arms {grounded mix / no-ontology ablation /
  standard-only} — with a 2-factor **α×β** cell (α = ontology-corpus
  fraction, β = SQL/DDL fraction) replicated across rungs. *Eval:* a
  **FLOOR** (grounded ≈ standard on general LM evals → non-degeneracy)
  AND a **LIFT** (grounded > ablation on relational + de-novo
  Data-Element elucidation, cells-only / control-task / PR-metric,
  bootstrap CI). It also **calibrates** the cheap proxies (R1 /
  coverage-close / corpus-quality) against the pretrain signal.
  *Preconditions, both MET:* the corpus's max non-repetitive token yield
  is measured (Y_eff ≈ 7.93M byte-tokens, which caps α at scale), and
  the discriminating relational eval is **realization-as-CPA** — per
  chapter, cited templates → a Domain/Range TBox, corpus tables → an
  ABox, HermiT *realizes* each column's type (CPA by a sound-and-complete
  oracle), with a Domain/Range-ablated matched-token control (pilot
  selectivity 0.79, 95% BCa CI [0.63, 0.88], permutation p = 1e-4 →
  CI-clean).
- **M3 — scale + the FINAL PHASE GATE. UNTESTED.** Conditioned on M2:
  climb to RWKV-7-matched params, extrapolate α*(N)/β*(N) to target
  scale, and confirm the lift persists.
- **M4 — the forward door (unlocked by the final gate).** Iterate the
  machinery: close the generator loop (SAE-instrumented Qwen with
  process-reward fine-tuning — the anti-Goodhart reward on the
  ontological-reasoning circuit, not the verdict), and apply the
  pipeline to a **novel domain** with minimal re-tuning (RASE
  generalization).

### The final phase gate

> At RWKV-7-matched scale, the ontology-grounded data mix yields an
> H-Net+RWKV model that **(a) matches RWKV-7 on general/standard evals**
> — the non-degeneracy floor — **AND (b) exceeds the no-ontology-ablation
> control on relational understanding + de-novo Data-Element
> elucidation**, CI-clean, with the α×β interaction and the mix-optimum
> scale-drift characterized.

Passing certifies the ontology machinery as a **valid instrument for
relational domain adaptation** and authorizes the M4 forward door.
Failing localizes the break to a *named arrow* — ontology→corpus
degeneracy (M2 floor), corpus→model transfer (M2 lift), or scale-drift
(M3) — each with its own remediation. This gate **supersedes** the
proxy-only corpus-as-deliverable gate: the proxies are *calibrated by
it*, never trusted ahead of it.

### Next moves (July 2026)

With both corpus-side phase gates passed, the near-term ladder is:

- **Path A training runs** (`src/aegir/flows/path_a_training_flow.py`)
  — the World-v3 continued-pretraining harness that isolates the
  *data* value of the generated corpus holding architecture fixed.
- **M1 — the H-Net isolation gate** (above) — the model track's next
  green light; blocks M2.
- **The latent-lens ladder.** Supervision for it now accumulates by
  operation: every topic-layer association record is a training pair
  for the anchor-projection head (L3), and lens identity fields are
  already stamped on every record and collection state (L1 — done).
  L2, the latent-prediction auxiliary for Path-B pretraining, is the
  next major move on this ladder.

## Shared infrastructure

Both artifacts depend on shared substrate beyond the ontology:

- **Gateway, leaderboard, and lineup.** The FastAPI gateway + React UI
  (above), plus the **lineup / KB** — the LINEUP navigation primitive
  (Ward Cunningham, credited; not a wiki — no editing/forking) over the
  KB, which is a build **projection** of the three Data Products
  (ontology / relational / content) at `build/dev/{current,scratch,archive}`
  (`src/aegir/lineup/`, `just kb-build`). Roots are refs: **current** =
  the latest `sdg-corpora` release as a complete kasten, **scratch** =
  trunk (the full live projection), **archive** = past releases and
  frozen snapshots; promotion snapshot-freezes current into archive
  while trunk rolls on. Topic-layer associations project as walkable
  item notes (Scratch → Content → Items × Topics → item → term). The
  read surface for run sidecars, ontology-rigor metrics, and
  corpus-quality surfaces alike.
- **The strategy pillar.** The `sdg-strategy` submodule (`strategy/`)
  is the "how it was made" Data Product — four pillars (lens / voices /
  knobs / targets) under a Merkle `strategy_id`
  (`src/aegir/strategy/manifest.py`), shadows-as-branches, promotion by
  cherry-pick. Every run-zettel cites its `strategy_id`; strategy drift
  at flow start is a hard failure.
- **Lineage substrate.** The Atlas-on-AGE provenance graph, with
  **OpenLineage / Marquez compatibility** and **Atlas deep integration**
  — implemented in full. The discipline that keeps it non-dependent: the
  ontology is the single source of truth, so Atlas and every projection
  stay rebuildable-from-the-ontology-or-it's-a-bug, never a master
  (Atlas edits *suggest*, never *commit*). See the Signals Programme's
  source-of-truth diagram.
- **The engine and meta-harness.** A gRPC engine serving
  Qwen3.6-35B-A3B-FP8 via vLLM under strict layering (engine→vLLM,
  workloads→gRPC), and the agent-mediated RETE/FSM control spine
  (`src/aegir/meta_harness/`) that orchestrates membrane-gated proposal.
- **Worktree-aware development tooling.** `git worktree`-based
  cross-checkout dev with shared `.git` and per-worktree service gating;
  the cross-worktree SAE-feature streaming pipe lives here. See
  [Worktree Aware Development](./worktree_aware.md).

## Design principles

1. **Each milestone produces a usable artifact.** M0 produced the
   substrate machinery (reasoner gates, evolving harness); the model
   track produced a training loop with leaderboard, an air-gap-deployable
   gateway, and a pretrained checkpoint; the ontology is itself a
   HermiT-validated, OQuaRE-GREEN, citable deliverable independent of any
   downstream model result.

2. **Empirical gates are real gates, not aspirations.** No scaled spend
   proceeds without a green gate. The OQuaRE publish gate refuses
   `sync --push` of the ontology below GREEN; the M1 isolation gate
   blocks M2; the M2 floor-and-lift gates M3; the M3 final gate
   authorizes the forward door. A gate that honestly held RED (the
   ontology rigor gate, before the rigor-evolution loop closed it) is the
   evidence that the gates bind.

3. **Locked artifacts are hash-tracked end-to-end.** Every run records
   its catalog version, locked-weights/null-statistics hashes, and run id
   in sidecar metadata; a strict-resume policy refuses to resume any run
   whose locked artifacts have drifted. Pipeline runs additionally pin
   the Merkle `strategy_id` and the topic-registry collection sha —
   every topic association carries the collection state that
   adjudicated it.

4. **Outward contracts stay narrow.** The project publishes the
   `sdg-corpora` SHARE tier (ontology + SKOS vocabulary + DDL spine +
   corpus), trained checkpoints, and (when stable) the SAE feature
   dictionary. Consumers' internal architectures (DST fusion, FSM session
   state, governance pipelines) are not the project's concern; this
   decoupling is what makes each artifact shippable in isolation.

5. **Complexity is bounded.** Each milestone adds exactly one new
   dimension of complexity (1-DOF where separable; a 2-factor cell only
   where the interaction *is* the hypothesis). Failure modes that respect
   this discipline are easy to diagnose; an honest revision pass on this
   document is the only protection against drift.

## Long-horizon work

The [agent swarm](./agent_swarm.md) modules in `src/aegir/swarm/` are
scaffolding for the long-horizon multi-agent training task; no operational
training uses them yet. The four-phase K2.5 PARL roadmap (supervised →
reward → PARL → swarm RL) folded into the Signals milestone ladder, and
the RL approach evolves through the GEPA / Agent-Lightning layers; the
reward-modeling, PARL-training, and swarm-RL design notes remain in the
tree as that design record.
