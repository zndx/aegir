# Signals Programme — Relational Domain Adaptation (aegir workstream)

Status: programme charter (2026-06-16). The go-forward development programme for the aegir track,
framed as a workstream of the cross-repo **Signals** initiative. Builds on
[`end_to_end_and_meta_harness.md`](./end_to_end_and_meta_harness.md) (the substrate-evolution machinery)
and the [`EVIDENCE.md`](../../../EVIDENCE.md) gate discipline. Designed to be lifted into a GitHub project
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
information domain with *minimal re-tuning* and measure what breaks (topic model, family complex, BFO
anchoring, R1). Promotes "valid instrument" → "validated method."

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
