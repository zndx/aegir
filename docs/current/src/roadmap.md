# Roadmap

The project pursues two coupled research outputs that share substrate
and are cited together. Each produces its own milestone progression
and its own evaluation surface.

- **Track 1 — Byte-level model for column annotation.** A hierarchical
  byte-level sequence model trained on a mixed corpus and fine-tuned
  for Column Type Annotation, Column Property Annotation, and
  cross-table data element discovery. Application paper.
- **Track 2 — RLVR for ontology generation.** A GRPO-trained policy
  that emits OWL ontology compositions scored by a four-component
  deterministic verifier *R(O, I)*. Research paper.

The two tracks share the SDG ontology, the procedural template
catalog, and the verbalization pipeline. They are coupled downstream:
Track 2's verifier-passing compositions become a corpus that Track 1
can pretrain on. The [concept brief](./ontology/concept_brief.md)
documents the two-paper structure in detail; the [semantic-engine
authoritative reference](./ontology/production_state.md) describes
the shared substrate and the in-flight empirical test of Track 2.

## Track 1 — Byte-level model + column annotation

### M0 — End-to-end training pipeline

Delivered. End-to-end BDD-backed training on the `gt-signals-dbpedia`
benchmark (120 DBpedia labels, 814 tables). Boundary diagnostics
visible per epoch. Checkpoint discipline (`outputs/runs/{run_id}/…`)
with sidecars and pre-rendered Bokeh plots.

### M1 — Air-gap leaderboard envelope

Delivered. FastAPI gateway on port 8091, React UI, ABI-patched
flash-attn / mamba-ssm build path. Deployment targets: devenv
(local), CAI (PGlite), Zarf air-gap K8s. The leaderboard reads
`outputs/runs/` directly — no tracking daemon, no W&B, no MLflow.

### M2 — Pretraining baseline + ontology ownership

The v2 mixed-corpus pretrain (2026-04-27) is the project's first
real backbone: 122k training steps on a 2 GB mixed corpus
(FineWeb-Edu + SQaLe + SchemaPile + FinePDFs-lab), single GPU,
≈10 h wall clock. Stratified held-out evaluation shows non-
degenerate representations across all four trained-time slices and
≈2 bpb drops on domain-targeted data with general prose held flat.
See [Training Regime §10](./training_regime.md) for the full table.

The remaining M2 scope is the v2 → SOTAB head fine-tune that closes
the loop on the 2026-04-19 representation-collapse incident. The
gate is documented in the [Ontology Charter § empirical
gate](./ontology/charter.md#empirical-gate-before-any-vocabulary-expansion):
≥ 3 MCL clusters, ≥ 0.10 macro F1, ≥ 10 distinct predicted labels.
No vocabulary expansion ships before that gate is green. Companion
work in this milestone:

1. Ontology + synth migration scaffolding — `src/aegir/ontology/`
   and `src/aegir/synth/` directories, SPARQL totality queries, CI
   gates.
2. `_LABEL_DIMS["sotab"]` reconciliation from the stale 91 to the
   verified 82 distinct SOTAB v2 Schema.org CTA labels.
3. `label_to_iri()` resolver consuming `vocab_label_map.json`.
4. `vocab_label_map.json` v1.0.0 — versioned outward contract,
   built deterministically from the TTL via SPARQL.
5. External-baseline harness — Nemotron 3 Nano, OpenAI OSS 20b,
   REVEAL re-implementation, run side-by-side on the leaderboard.
6. Per-class F1 bars in the leaderboard UI.

The [Supervised Bootstrapping page](./roadmap/supervised.md) expands
on the rationale for fine-tuning from a pretrained checkpoint
rather than training from random initialization, and the exact
thresholds the liveness gate measures.

### M3 — Vocabulary expansion + multi-GPU step-up

Conditional on M2 clearing the liveness gate. Components likely in
scope:

1. Multi-GPU pretraining at the next byte-budget bump (8 GB on
   6 × RTX 4090 ≈ 7 h, vs. v2's 10 h on a single GPU at 2 GB).
2. A v3 corpus mix that may incorporate verifier-passing synthetic
   slices from Track 2's policy (see Track 2 below). v2 baseline
   thresholds: keep `eval.fineweb-held` ≤ 1.61, push
   `eval.finepdfs-lab-held` below 1.78, no regression on
   SchemaPile or SQaLe.
3. BIRD held-out as a second transfer probe (Spider already in
   v2 as transfer-from-SQaLe).
4. KServe `InferenceService` predictor for online inference on
   Cloudera AI Inference Service.
5. GPU-flavored Zarf image bundling a trained checkpoint.

### M4 — Production CTA/CPA evaluation

Competitive scores against published baselines on SOTAB-CTA,
GitTables, WikiTables. Requires a non-degenerate v2 → SOTAB head
fine-tune (M2) and likely the multi-GPU step-up at `base` config
(≈ 500M params, M3). Target F1s are deferred until the empirical
gate clears.

## Track 2 — RLVR for ontology generation

### Catalog and verifier infrastructure

Delivered. The SDG ontology has been authored across seven batches
of templates (Batches 1–7, totaling 540 Manchester-syntax axiom
templates) with 522 verbalized (97% coverage) and 485 flagged as
`is_complex` (90%). The four-component deterministic verifier
*R(O, I)* is operational with aggregation weights locked at
`{a, b, c} = {0.50, 0.05, 0.45}` from a 30-ontology hand-authored
discrimination sweep (AUC 0.9956, mean *R*-separation 0.336). The
held-out 50 evaluation set (25 good + 25 bad scenarios authored
before any policy-side RL work began) gives separation 0.5129
against the locked verifier. The verifier is deterministic,
hash-stable, and has no JVM dependency in its runtime hot path.

### Policy training (in flight)

A GRPO-trained policy on Qwen3.5-9B-Base, with a LoRA adapter on
attention and MLP projections and the corresponding
`SAE-Res-Qwen3.5-9B-Base` residual-stream adapter held untouched
for interpretability. The training pipeline includes:

1. Constrained-decode JSON Schema enforcement via lm-format-enforcer,
   wired through a wrap on `model.generate` that survives TRL's
   `unwrap_model` indirection.
2. A rejection-sampling SFT bootstrap that warm-starts the policy
   from rejection-sampled high-reward Base-model outputs (Option A
   in the
   [authoritative reference's empirical-test section](./ontology/production_state.md#6-the-current-empirical-test-under-revision)).
3. Per-iteration verifier scoring with the locked *R(O, I)* and
   group-relative advantage estimation.

The current run is the first end-to-end empirical test of the
warm-start procedure. The methodological choice between Option A
(rejection-sampling SFT), Option B (Instruct variant with
Instruct-paired SAE), and Option C (Self-Distillation Fine-Tuning)
is
[explicitly under revision](./ontology/production_state.md#62-three-candidate-warm-start-procedures);
the in-flight run will settle whether Option A is sufficient.

### Held-out evaluation and paper 1

The first held-out evaluation of the GRPO-trained policy against
the held-out 50 is the gate for the paper-1 claim
(C2 — optimizability). A direct comparison against Option B is the
next-priority experimental step once Option A produces an initial
result, and is necessary to settle the warm-start choice on grounds
other than "the in-flight run worked or did not."

Paper 1's contribution is the verifier *R(O, I)* and the GRPO loop
that targets it. C1 — discrimination on the C1 test set — has been
established. C2 — optimizability of *R* via GRPO — is under test.
The two together form paper 1's headline.

### Paper 2 — ontology-grounded byte-level pretraining

Paper 2's claim is downstream: verbalizations from *R*-passing
ontologies (produced by Track 2's policy) measurably improve
byte-level pretraining on the Track 1 evaluation surface. Paper 2
is contingent on Track 2's policy producing verifier-passing
compositions at corpus scale. Its methodology will be refined after
paper 1's results constrain it.

## Shared infrastructure

Both tracks depend on shared substrate beyond the SDG ontology:

- **Gateway and leaderboard** (M1) — FastAPI gateway, React UI,
  cross-worktree-aware deployment; the read surface for both
  Track 1's run sidecars and Track 2's GRPO metrics.
- **Lineage substrate** — Atlas-canonical entity / process /
  Business-Glossary projection with an OpenLineage-compatible push
  to Marquez for broader-stack consumers. Every training run leaves
  a synchronous WAL plus an asynchronous Atlas projection; Track 2's
  verifier metrics flow through the same path.
- **Worktree-aware development tooling** — `git worktree`-based
  cross-checkout dev with shared `.git` and per-worktree service
  gating (Postgres, Qdrant, gateway, vite-dev). The cross-worktree
  SAE-feature streaming pipe lives here. See
  [Worktree Aware Development](./worktree_aware.md).

## Deferred work

The [agent swarm](./agent_swarm.md) modules in `src/aegir/swarm/`
are infrastructure scaffolding for a future multi-agent training
task. An earlier four-phase K2.5 PARL roadmap (supervised → reward
→ PARL → swarm RL) has been superseded by the two-track structure
above; the swarm modules remain in the codebase as infrastructure
but no operational training currently uses them. Three legacy
pages — Phase 2: [Reward Modeling](./roadmap/reward.md), Phase 3:
[PARL Training](./roadmap/parl.md), and Phase 4:
[Agent Swarm RL](./roadmap/swarm_rl.md) — describe that deferred
work and remain in the repository for archival continuity.

## Design principles

1. **Each milestone produces a usable artifact.** M0 produces a
   training loop with leaderboard. M1 produces an air-gap-deployable
   gateway. M2 produces a pretrained checkpoint and a versioned
   vocabulary. Track 2's catalog and verifier are themselves a
   citable artifact independent of the paper-1 policy result.

2. **Empirical gates are real gates, not aspirations.** The
   v2 → SOTAB liveness gate blocks M3 if it fails; the C1 sweep AUC
   gated the verifier weight lock; the held-out 50 separation gates
   the paper-1 claim. Vocabulary expansion against a collapsed model
   adds labels for predictions that aren't getting made; verifier
   weight relocking against a degenerate test set adds *R*-discrimination
   that isn't there.

3. **Locked artifacts are hash-tracked end-to-end.** Every run
   records the catalog version, locked weights hash, null-statistics
   hash, and run id in its sidecar metadata; a strict-resume policy
   refuses to resume any run whose locked artifacts have drifted.

4. **Outward contracts stay narrow.** The project publishes
   `vocab_label_map.json`, trained checkpoints, the SDG vocabulary
   TTL, and (when stable) the SAE feature dictionary. Consumers'
   internal architectures (DST fusion, FSM session state, governance
   pipelines) are not the project's concern; this decoupling is what
   makes each track shippable in isolation.

5. **Complexity is bounded.** Each milestone adds exactly one new
   dimension of complexity. Failure modes that respect this discipline
   are easy to diagnose; failure modes that don't are the hardest to
   spot, and an honest revision pass on this document is the only
   protection against drift.
