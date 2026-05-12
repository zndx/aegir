# Roadmap

Two tracks run in parallel. The **near-term milestone track** (M0–M4)
is the operational roadmap — what ships, in what order, against what
empirical gates. The **far-future RL post-training track** (Phase 1–4)
is the long-horizon architectural plan that the milestone track
eventually converges into.

The milestone track describes operational deliverables in their
delivery order. The RL track describes the longer-horizon
architectural plan that the milestone track is intended to converge
into.

## Near-term milestones

### M0 — End-to-end training pipeline

Delivered. End-to-end BDD-backed training on the real
`gt-signals-dbpedia` benchmark (120 DBpedia labels, 814 tables).
Boundary diagnostics visible per epoch. Checkpoint discipline
(`outputs/runs/{run_id}/…`) with sidecars and pre-rendered Bokeh
plots.

### M1 — Air-gap leaderboard envelope

Delivered. FastAPI gateway on port 8091, React UI, ABI-patched
flash-attn / mamba-ssm build path. Deployment targets: devenv
(local), CAI (PGlite), Zarf air-gap K8s. The leaderboard reads
`outputs/runs/` directly — no tracking daemon, no W&B, no MLflow.

### M2 — Pretraining baseline + ontology ownership

**Empirical anchor: v2 mixed-corpus pretrain (2026-04-27)**

The v2 byte-level pretrain is the project's first real backbone. 122k
training steps on a 2 GB mixed corpus (FineWeb-Edu + SQaLe + SchemaPile
+ FinePDFs-lab), single GPU, ~10 h wall clock. Stratified held-out
eval shows non-degenerate representations across all four trained-time
slices and ~2 bpb drops on domain-targeted data with general prose
held flat. See [Training Regime](./training_regime.md) §10 for the
full table; the result establishes that the architecture is learning
real structure across all four stratified-eval slices rather than
exploiting an easier distribution.

**M2 scope (in progress):**

1. **Validate v2 → SOTAB head fine-tune produces non-degenerate
   per-class F1** — the gating risk that closes out the April
   representation-collapse incident. See
   [Ontology Charter §empirical gate](./ontology/charter.md#empirical-gate-before-any-vocabulary-expansion)
   for the precise thresholds (≥ 3 MCL clusters, ≥ 0.10 macro F1, ≥ 10
   distinct predicted labels). **No vocabulary expansion ships before
   this gate is green.**
2. **Ontology + synth migration scaffolding** — `src/aegir/ontology/`
   + `src/aegir/synth/` directories created, vocabulary copied from
   the sibling project, SPARQL totality queries authored, CI gates
   wired. See [Ontology Migration](./ontology/migration.md) for
   phased plan.
3. **`_LABEL_DIMS["sotab"] = 91 → 82` reconciliation** in
   `src/aegir/data/table_dataset.py`. The SOTAB v2 Schema.org CTA CSV
   union is verified at 82 distinct labels.
4. **`label_to_iri()` resolver** consuming `vocab_label_map.json`. The
   smallest end-to-end vertical slice that proves the pipeline.
5. **`vocab_label_map.json` v1.0.0** — versioned outward contract,
   built deterministically from the TTL via SPARQL. Only released
   after the M2.1 empirical gate is green.
6. **External-baseline harness** — Nemotron 3 Nano, OpenAI OSS 20b,
   REVEAL reimpl, run side-by-side on the leaderboard.
7. **Per-class F1 bars** in the leaderboard UI.

### M3 — Vocabulary expansion + multi-GPU step-up

**Conditional on M2 completing.** Vocabulary expansion past the
copied baseline (the bespoke vocab in its current form) is scoped
against Ægir's empirical priorities, not against an externally-defined
tier breakdown. See [Ontology Migration §Phase 6](./ontology/migration.md#phase-6--vocabulary-expansion-gir-defined-not-atelier-tiered).

Likely components:

1. Multi-GPU pretraining at the next byte-budget bump (8 GB on 6 ×
   RTX 4090 ≈ 7 h, vs. v2's 10 h on a single GPU at 2 GB). The DDP
   path is already proven; what's new is the budget.
2. A v3 corpus mix that may or may not include ontology-conditioned
   synthetic slices, depending on M2 outcomes. v2 baseline thresholds
   are: keep `eval.fineweb-held` ≤ 1.61, push
   `eval.finepdfs-lab-held` below 1.78, no regression on
   schemapile/sqale.
3. BIRD held-out as a second transfer probe (Spider already in v2 as
   transfer-from-SQaLe).
4. KServe `InferenceService` predictor for online inference on
   Cloudera AI Inference Service.
5. GPU-flavored Zarf image bundling a trained checkpoint.
6. Datashader/Dask for large-run visualization.

### M4 — Production CTA/CPA evaluation

Competitive scores against published baselines on SOTAB-CTA,
GitTables, WikiTables. Requires a non-degenerate v2→SOTAB head
fine-tune (M2) and likely the multi-GPU step-up at `base` config (~500M
params, M3). Target F1s deferred until the empirical gate clears —
quoting numbers before the model demonstrably escapes representation
collapse on the supervised objective is premature.

## Far-future: K2.5 RL post-training

This section outlines the four-phase plan for training Ægir from a
supervised baseline through full multi-agent reinforcement learning
with PARL orchestration. The supervised baseline at the bottom of this
plan is what M4 produces; the RL phases come *after* that.

The training follows a progressive complexity increase, where each
phase builds on the previous one's checkpoints and infrastructure:

```
Phase 1              Phase 2              Phase 3              Phase 4
Supervised     -->   Reward         -->   PARL           -->   Agent
Bootstrapping        Modeling             Training             Swarm RL

Train base           Design reward        Train orchestrator   Scale to
Aegir on CTA/CPA     components and       with frozen          multi-specialist
benchmarks           validate signals     specialists          swarms
```

### [Phase 1: Supervised Bootstrapping](./roadmap/supervised.md)

The current empirical reality has narrowed Phase 1's scope. The
original Phase 1 plan ("train from random on CTA/CPA") was invalidated
by the Apr 19 representation collapse. Phase 1 today means:
**v2 byte-level pretrain → CTA/CPA head fine-tune from that
checkpoint → competitive F1**. The pretrain is done; the fine-tune is
the M2 gate.

### [Phase 2: Reward Modeling](./roadmap/reward.md)

Design and validate the three reward components (`r_perf`,
`r_parallel`, `r_finish`) that will drive PARL training. Calibrate
lambda weights and verify that the reward signal produces meaningful
gradients. Cannot start until Phase 1 produces a non-degenerate
checkpoint.

### [Phase 3: PARL Training](./roadmap/parl.md)

Freeze the best Phase 1 checkpoint as a specialist and train a new
primary model with the PARL orchestrator. Use token-level clipping RL
with critical-steps optimization.

### [Phase 4: Agent Swarm RL](./roadmap/swarm_rl.md)

Scale from a single specialist to a full swarm with dynamic specialist
spawning. Implement wide search (parallel column analysis) and deep
search (hierarchical type reasoning) patterns.

## Design principles (apply to both tracks)

1. **Each milestone produces a usable artifact.** M0 produces a
   training-loop with leaderboard. M1 produces an air-gap-deployable
   gateway. M2 produces a pretrained checkpoint and a versioned
   vocabulary. Even far-future Phase 1 yields a competitive standalone
   column annotation model.

2. **Empirical gates are real gates, not aspirations.** v2 → SOTAB
   head fine-tune liveness checks block M2.5; if they fail, M3 waits
   until the architecture is debugged. Vocabulary expansion against a
   collapsed model adds labels for predictions that aren't getting
   made.

3. **Frozen specialists are never modified.** PARL training only
   updates the primary model and the routing/fusion modules. This
   prevents catastrophic forgetting in specialists and simplifies the
   training loop.

4. **Reward components are validated independently.** Phase 2 exists
   specifically to ensure that `r_parallel` and `r_finish` produce
   meaningful gradients before combining them with `r_perf` in Phase
   3.

5. **Complexity is additive, not multiplicative.** Each phase adds
   exactly one new dimension of complexity (multi-task → reward signals
   → RL policy → multi-agent), making failures easy to diagnose.

6. **Outward contracts stay narrow.** Ægir publishes
   `vocab_label_map.json` + checkpoints. Consumers' internal
   architectures (DST fusion, FSM session state, governance pipelines)
   are not Ægir's concern. This decoupling is what makes the project
   shippable in isolation.
