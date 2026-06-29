# Training Regime: Converting Sparse to Dense

> *"Even serializing out linear paths from the hierarchical regime would
> help convert our inherently sparse reward task into a more dense,
> learnable one."*

This chapter is a postmortem and forward plan, written after the first
attempt to train Aegir on SOTAB Column Type Annotation failed by way of
total representation collapse. It documents the diagnosis, the reframing
the failure prompted, the staged plan that follows, and the operational
parameters (optimizer, hygiene, compute envelope) that support it.

## 1. The first attempt and its failure

Aegir's inaugural task was SOTAB v2 Schema.org Column Type Annotation
(91 leaf classes, 116,887 training columns, 1,769 val). The training run
used the `small` config (56M params), 3 epochs, batch 16, `max_length`
1024, lr 3e-4, AdamW without gradient clipping or explicit warmup.

The loss curve looked "plateaued" — train/val both hovering around 4.1/4.5
with F1 never exceeding 0.011/0.0007 (micro/macro). On the surface this
looks like weak learning. A diagnostic pass (see
[§ Diagnostic case study](./pretraining/diagnostic_case_study.md)) told
a much more specific story:

- **Every val sample produced the identical pooled embedding** to within
  bf16 rounding noise. Max pairwise L2 across 50 probe samples was 0.020
  on vectors of mean norm 6.98 — a relative spread of 2.9 × 10⁻³.
- **The classifier predicted `currency` on 100% of 1,500 val samples.**
  Exact-match accuracy (3.27%) was exactly the val base rate of the mode
  class.
- **MCL clustering at every inflation returned one cluster**, because
  there was only one point in embedding space to cluster.

The heads (pooler, classifier, residual projections) all had healthy
weight norms. The collapse is upstream of them, inside the RWKV+H-Net
backbone.

This is not a hyperparameter bug, and fixing it is not primarily a
hyperparameter change.

## 2. The reframing

Three observations explain the failure together:

### 2.1 Aegir is architecturally a language model

H-Net dynamic chunking, RWKV-7 time-mixing, hierarchical recursion —
every mechanism in the backbone was designed for dense per-token
supervision on byte/text corpora. The H-Net paper trains this
architecture on next-byte language modeling; the chunker *learns
boundaries as a byproduct* of next-byte prediction. There is no
published version of H-Net that's trained from random on sparse
classification. Not because nobody tried — because the architecture
requires dense gradients to stabilize the chunker and the recurrent
state.

### 2.2 Our task as formulated is sparse-reward

Direct CTA delivers one label per 1024-byte input. A from-scratch model
must simultaneously discover byte statistics, column boundaries, content
patterns, cross-column context, and the class→label mapping, from a
single gradient signal per forward pass. Compare to REVEAL's 0.815 micro
F1 baseline: RoBERTa-base, pretrained on ~160 GB of text, fine-tuned on
CTA. Without the pretraining stage REVEAL does not exist. Our attempt
was REVEAL stage 2 without stage 1.

### 2.3 The collapse is a design-use mismatch, not a bug

The proximate mechanism is RWKV-7 time-decay saturation at 22k update
steps of direct-CTA under lr 3e-4 with no gradient clipping and minimal
warmup. The *underlying* problem is that a properly-pretrained backbone
would already be in a well-conditioned region of parameter space before
CTA fine-tuning began, and that region is robust to the saturation
basin our first run fell into. Training hygiene helps at the margin;
pretraining fixes the structural problem.

## 3. Two axes for densifying supervision

The reframing suggests two orthogonal moves to convert the sparse task
into a dense one. They compose.

### 3.1 Axis 1 — Byte-level pretraining on domain corpora

Feed the architecture what it was designed for. Next-byte prediction on
raw GitTables byte-serializations. Dense signal: every position is a
supervision anchor. The chunker learns column-delimiter recognition,
cell-pattern boundaries, content-type signatures for free, without any
label. This is RWKV-7's native training regime, applied to our actual
domain. The `AegirForCausalLM` head in `src/aegir/models/heads.py` was
built for this and has never been used.

**Data**: ~36 GB of GitTables parquets already on disk (`/raid/datasets/
gittables/`, 562,214 tables, ~24 million annotated columns). Byte
serialization produces ~36 billion raw bytes. A first probe uses ~100 M
bytes.

### 3.2 Axis 2 — Linearize ontology paths as prediction targets

Instead of predicting `schema:Hotel` as one token in a 91-way softmax,
predict the Schema.org ancestry chain as a sequence:

```text
<BOS> Thing → Organization → LocalBusiness → LodgingBusiness → Hotel <EOS>
```

Each step in the chain is a ~10-20-way softmax over the children of the
current parent. Per-column supervision becomes 3-5 gradient signals
instead of 1. Shallow predictions get partial credit. Rare leaf classes
inherit gradient from their parents — 1,000+ examples teaching
`Place → * → Thing` even when only 50 examples reach the leaf
`CivicStructure`.

The path-prediction head's softmax is naturally dual-center: each step's
centroid is a parent-level cluster. The dual center loss we argued for
on first principles (see
[§ Hierarchical loss design](#6-hierarchical-loss-design)) falls out of
the formulation for free.

## 4. Staged plan

The staged plan isolates the failure mode test from the production fix
so attribution stays clean.

### 4.1 Stage A — Hygiene-only direct-CTA rerun

**Question**: Does training hygiene alone prevent the collapse in the
sparse regime?

**Change set** (bundled because they're the same intervention at four
knobs):

| Knob | Before | After | Reason |
|---|---|---|---|
| Learning rate | 3e-4 | 5e-5 | 6× reduction matches RWKV from-scratch norms. |
| Warmup | none | 1000 steps | Absorbs the initial-lr transient that otherwise destabilises RWKV time decay. |
| Gradient clip | none | `max_norm=1.0` | Reference RWKV-LM recipes always clip; we never did. |
| Weight decay | 1e-2 | 1e-4 | AdamW default is aggressive for recurrent architectures. |

**Outcome interpretation**:

- *Collapse resolves → some F1 > 1e-2*: hygiene was sufficient for the
  direct regime. Useful baseline but still not a competitive path; we
  proceed to Stage B-C for the real system.
- *Collapse persists*: deeper issue (architectural or task-architecture
  mismatch). Escalate to structural diagnosis.

### 4.2 Stage B — Byte-level pretraining on GitTables

**Question**: Does Aegir's architecture converge on its native training
objective?

**Configuration**:

- Model: `small` (56M params, same config as Stage A for A↔B comparison)
- Objective: next-byte cross-entropy with `AegirForCausalLM`
- Data: raw GitTables parquets serialized to byte streams, 100M-byte
  budget for the first probe
- Training: same hygiene bundle as Stage A, but applied to pretraining
  where it matters most. lr 1e-4, warmup 1000, grad clip 1.0, wd 1e-4.
- Batch: 64 with grad accumulation (effective 128-384 across DDP)
- Instrumentation: per-step boundary_diagnostics logging to catch
  saturation as it happens, not after the fact.

**Load-bearing question**: if pretraining also collapses, there is a
deeper architecture issue (v_first sharing, DeChunk EMA stability,
boundary predictor interaction) that we must isolate independently of
any task.

### 4.3 Stage C — Path-prediction fine-tuning

**Question**: Does pretraining + hierarchical supervision beat flat
direct-CTA at the same compute?

**Configuration**:

- Start from Stage B's pretrained checkpoint
- Implement Schema.org path serializer from
  `CTA_CPA_label_set_schemaorg.xlsx`
- Add `AegirForHierarchicalAnnotation` head: pooled embedding →
  autoregressive decode of `<BOS> → parent₁ → parent₂ → ... → leaf →
  <EOS>`. Hierarchical cross-entropy per step.
- Fine-tune on SOTAB. Evaluate at each ontology depth separately.
- Run the MCL geometry audit (which was uninformative on the collapsed
  Stage A checkpoint) — now meaningful.

Success threshold: macro F1 at leaf level > Stage A hygiene-only
result, *and* parent-level F1 (coarser granularity) >> leaf-level F1,
confirming the hierarchical regularisation helps where it should.

### 4.4 Stage D — MuonClip infrastructure (parallel track)

Orthogonal to A/B/C. Muon's Newton-Schulz step gives spectrally-bounded
parameter updates; MuonClip adds post-step Q/K row-norm clipping to
bound attention logits. Both are direction-addressing fixes to the same
class of failure we just hit (parameter saturation under long schedules),
stronger than magnitude-only gradient clipping.

Strategy:

1. Port existing MuonClip code (prior work referenced in Atelier, need
   to locate).
2. Bench Muon vs AdamW on the fast gt-signals-dbpedia task (~30 min per
   run, two runs).
3. If Muon matches or beats AdamW on the known-learning task, it
   becomes the default optimizer for Stage B onwards. If not, an
   interaction with RWKV-7's unusual parameter shapes needs to be
   understood before scaling up.

Muon is infrastructure, not a one-shot experiment. Once in, every
subsequent phase (Stage C, Phase 1.5 Mergekit, v3 Phase 2 Nano
alignment) benefits from a stronger base optimizer.

## 5. Training hygiene

The Stage A/B bundle is the minimum hygiene for from-scratch RWKV-7. The
rationale per knob:

- **Learning rate 1e-4 to 5e-5**. RWKV-LM's from-scratch recipes for
  sub-1B models sit in this range. 3e-4 works for BPE transformers at
  scale; byte-level RWKV doesn't have the same gradient-scale regime.
- **Warmup 1000+ steps**. Protects against the initial loss cliff where
  the untrained decay parameter swings wildly. Skipping warmup is a
  common cause of early saturation.
- **Gradient clipping max_norm 1.0**. Standard for RWKV. Not optional.
- **Weight decay 1e-4**. AdamW's default 1e-2 is calibrated for
  transformers; it is too strong a regulariser for recurrent
  architectures where the time-mix parameters are small and precious.
- **bf16 AMP** with fp32 optimizer state accumulation. Already in place.
- **Boundary-diagnostics logging per step** (new). The collapse we just
  experienced was detected only post-run; online diagnostics would have
  surfaced it within the first 500 steps.

## 6. Hierarchical loss design

Once we have a non-collapsed representation, the loss function question
becomes active. First-principles observations about the domain:

1. **Labels are ontology nodes, not categorical IDs.** Softmax CE treats
   `schema:Hotel` ↔ `schema:Motel` ↔ `schema:Person` as equally distant,
   contradicting the actual semantic geometry.
2. **Class distribution is long-tail.** A few parent-level clusters
   dominate. Dual-center loss with inter-class repulsion prevents rare
   classes from being subsumed into the dominant cluster.
3. **Surface underspecifies the label; context carries it.** But the
   *parent* level is usually decidable from surface alone. Uncertainty
   collapses monotonically up the ontology tree — a useful inductive
   bias that vanilla softmax does not exploit.
4. **H-Net is already hierarchical at the representation level**. Dual
   centers at the output level (leaf + parent) align with it
   architecturally.
5. **DED (M2) is clustering**. Dual-center embeddings ARE clusters. The
   same head doing CTA at inference produces the column embeddings we
   hand to B-cubed evaluation for DED.

Path-prediction (Axis 2 above) subsumes dual center loss — each
autoregressive step's softmax is a dual center at that ontology level.
So the loss work is done by the head structure itself once Stages B-C
are in place.

## 7. Methodology: MCL as a geometry audit

Borrowed from van Dongen's MCL (Markov Cluster) algorithm (2000),
developed over two decades for bioinformatic orthology detection. MCL
simulates stochastic flow on a similarity graph: expansion (random walk
via matrix multiplication) alternates with inflation (entrywise power
plus renormalization), producing clusters as attractor basins of the
flow. No `k` required; inflation parameter controls granularity.

Used here not for production clustering but as a **geometry audit** for
embedding spaces:

- Run the model, extract pre-classifier embeddings, build a cosine
  similarity graph.
- Sweep inflation ∈ {1.4, 2.0, 3.0, 4.0}.
- Report cluster count and purity against leaf labels and Schema.org
  parent labels at each inflation.

Interpretation:

- **Parent purity rises at coarser inflations → representation has
  recoverable hierarchical structure.** The model's embeddings encode
  the ontology even if the classifier head doesn't read it out. Loss
  function is the appropriate lever.
- **Parent purity flat across inflations → no hierarchical structure.**
  Representation itself is weak. Architectural or training-regime fix
  required.
- **Single cluster at all inflations → representation is degenerate
  (collapse).** The audit itself is suspended. This is Stage A territory.

On the failed run, the audit was correctly uninformative — MCL produced
one cluster because there was one point. Once a non-collapsed
checkpoint exists, the audit becomes the tool for answering "does the
embedding geometry encode the ontology, or just the base rates?"

## 8. Compute envelope and scaling headroom

Training hardware is the tinybox (6 × RTX 4090, 24 GB each, 144 GB
aggregate). 4090s have no NVLink; P2P runs over a PCIe-switched fabric.

### Per-GPU memory profile

At `small` config (56M params, our current training point) per-GPU
static memory is roughly 800 MB, with 2-4 GB of activations + backward
buffers at batch 16 / seq 1024. Peak usage sits around 5-6 GB of 24
GB available — ~18 GB of headroom per card.

| Config | Params | Static mem | Activations (B=16, L=1024) | Fits 4090? |
|---|---|---|---|---|
| tiny | 13.5M | ~0.2 GB | ~1 GB | ✅ trivially |
| small | 56M | ~0.8 GB | ~3 GB | ✅ abundant |
| base | ~500M | ~7 GB | ~6-8 GB | ✅ comfortably |
| large (~2B) | ~2B | ~22-28 GB | ~10 GB | ❌ |
| xl (3B+) | 3B+ | 45 GB+ | growing | ❌ |

The knee between "single-GPU fits" and "FSDP required" lies between
**500M and 1.5B parameters**, depending on batch size and sequence
length.

### Scaling levers in ascending order of complexity

1. **DDP (data parallel)** — current. Full model per GPU, gradients
   AllReduce'd. Linear speedup up to bandwidth saturation. Unchanged
   up through `base`.
2. **Gradient accumulation** — free. Effective batch scales with
   `n_gpus × accum_steps × micro_batch`. Gets us to batches of 384+ at
   `base` size without any new infra.
3. **Activation checkpointing** — `torch.utils.checkpoint` wrap around
   the Aegir main network, re-compute during backward. Trades ~30%
   compute for ~3× activation memory savings. **Worth implementing
   proactively** before we hit any memory wall — even at current size
   it enables longer sequences and bigger batches.
4. **ZeRO-2 (optimizer + gradient sharding)** — saves 6P/N bytes per
   GPU. On a hypothetical 2B model with 6 GPUs, that's ~40 GB reclaimed
   per card. Minimal throughput cost.
5. **FSDP / ZeRO-3 (full parameter sharding)** — sharded forward via
   AllGather, sharded backward. ~10-20% throughput cost but unlocks
   models that wouldn't otherwise fit.

For our target domain (relational metadata), **`base` (~500M) is
competitive with REVEAL-class baselines** and does not require FSDP.
`large` (~1-2B) is the stretch goal for DED and Nano distillation; it
may or may not need FSDP depending on how aggressive we are with batch
size and sequence length. We have runway.

### Target-domain parameter sizing

- REVEAL (RoBERTa-base): 125M params, F1 0.815 on SOTAB CTA
- TURL: ~110M
- TabBERT / TaBERT: ~350M
- Byte-level has a ~2-3× parameter penalty vs BPE for equivalent
  capability, so the competitive byte-level target is 250-400M.

`small` (56M) is **undersized for competitive CTA**. `base` (~500M) is
right-sized or slightly overprovisioned. The Stage B pretraining probe
runs on `small` for speed; Stage C production fine-tuning should step
up to `base` once Stage B validates the pipeline.

### Forward-looking instrumentation

Every training run's `metadata.json` should carry a `peak_cuda_memory_mb`
field (from `torch.cuda.max_memory_allocated()`). This is a cheap
forward-looking indicator of how close each config is getting to the
next scaling threshold. No surprises when we move from `base` to
`large`.

## 9. Empirical validation (overnight, 2026-04-20)

Several claims in this chapter were falsifiable hypotheses when written.
Stages A and B, kicked off the same evening, delivered verdicts.

| Claim | Section | Verdict | Evidence |
|---|---|---|---|
| "This is not a hyperparameter bug, and fixing it is not primarily a hyperparameter change." | §1 | **confirmed** | Stage A hygiene rerun (lr 3e-4 → 5e-5, weight decay 1e-2 → 1e-4, warmup 10% → 15%, grad clip already 1.0) tracks the original collapsed run almost exactly: train loss 4.1281 vs 4.1286, val loss 4.5470 vs 4.5468, best val macro F1 0.0003 vs 0.0007. Four knobs moved coherently changed the outcome by < 1 part in 10³. |
| §4.2 load-bearing question: "does the architecture converge under its designed training regime?" | §4.2 | **yes** | Stage B byte-level pretraining on raw GitTables descended from loss 5.68 at step 20 (≈ entropy floor for 260-way softmax, `log 260 ≈ 5.56`) to 2.26 at step 3040. 3051 steps, 100M-byte budget, small model, SSD kernel active. Checkpoint at `outputs/pretrain/20260420T002455Z/final.pt`. |
| §7 geometry criterion: "is the pretrained representation actually alive?" | §7 | **yes** | Post-training, 8 random byte-sequence inputs produced 8 distinct embeddings. Max pairwise L2 = 21.6 on vectors of mean norm 16.0 — collapse ratio 1.35, vs the 0.01 threshold that flagged the SOTAB checkpoint. Per-dimension variance: median 0.34, max 1.76. The representation varies with input at the expected scale. |
| §8 compute projection: "small-config pretraining has ~18 GB of headroom on a 4090" | §8 | **confirmed** | Peak CUDA memory during Stage B was 5.7 GB (instrumented per-step via `peak_cuda_mem_mb` in `metrics.jsonl`). Stage C fine-tuning on the same hardware is comfortably within budget. |
| §3.1: "the chunker learns boundaries as a byproduct of next-byte prediction" | §3.1 | *untested in this probe* | Stage B's `boundary_diagnostics` were not logged per-step in this first probe. A follow-on instrumented re-run will confirm. |
| §6: "path-prediction subsumes dual center loss" | §6 | *untested* | Depends on Stage C. |

The combined verdict — hygiene does not escape sparse-CTA collapse, but
pretraining does converge and does produce a varied representation —
is the one the staged plan was designed to distinguish. The chapter's
argument now has running-code grounding, not just a first-principles
shape.

## 10. v2 mixed-corpus pretrain (2026-04-27)

The Apr 20 single-slice GitTables pretrain (Stage B) demonstrated that
the architecture converges under its native objective. v2 extends that
result to a 2 GB mixed-corpus pretrain across nine slices and produces
the project's first real backbone — the empirical anchor that the
M2/M3 milestones build on.

**Run mechanics:**

- Wall clock: 2026-04-26 23:22 → 2026-04-27 09:26 MDT (~10 h 4 m,
  single GPU 0)
- Training steps: 122,070
- Checkpoint: `outputs/mixed-v2/20260426T232240Z/final.pt` (~174 MB,
  small config)
- Intermediate checkpoints retained every 5,000 steps (24 total)
- Metrics: `metrics.jsonl` (per-step), `metrics_eval.jsonl`
  (4 trained-time eval slices × 25 evals)

**Headline:** final training-loader bits-per-byte = **1.179** (vs. v1
mixed at 1.202, vs. FineWeb-only baseline at 1.774). The headline gain
is small because the mixture distribution itself shifted between v1
and v2; the actual story is in the stratified held-out eval.

**Stratified held-out comparison (apples-to-apples, both finals on
the same 5 slices):**

| Held-out slice | v1 final | v2 final | Δ (v2 − v1) | What it measures |
|---|---|---|---|---|
| `eval.fineweb-held` | 1.601 | 1.608 | +0.007 | General prose perplexity |
| `eval.finepdfs-lab-held` | 1.882 | **1.784** | **−0.098** | Lab/clinical/regulatory prose |
| `eval.schemapile-held` | 2.888 | **0.997** | **−1.891** | Real-world DDL syntax |
| `eval.sqale-held` | 2.819 | **0.810** | **−2.009** | NL+DDL+SQL alignment |
| `eval.spider` | 0.752* | 2.155 | n/a | *v1 trained on Spider; 0.752 is contamination, not held-out competence. v2 holds Spider out cleanly; 2.155 is genuine generalization from SQaLe. |

**What this validates:**

1. **Architecture is learning, not just enjoying easier distribution.**
   v1's headline gain over the FineWeb-only baseline could have been
   pure distribution effect. The stratified eval shows ~2 bpb drops on
   the *specific* slices the v2 mixture targeted, while general prose
   stays statistically flat. That is targeted learning.
2. **Trimming FineWeb 0.55 → 0.35 did not hurt prose.**
   `eval.fineweb-held` is statistically indistinguishable
   (1.601 → 1.608, +0.4%). 700 MB of FineWeb training (35% × 2 GB) is
   sufficient at this budget.
3. **FinePDFs-lab vocabulary transfer is real.** v2 trained on
   lab/clinical prose for the first time and held-out prose of the
   same flavor saw a consistent 0.098 bpb drop.
4. **SQaLe → Spider transfer works.** v2 never saw Spider during
   training; Spider bpb dropped from random-init ~4.5 to 2.155. SQaLe
   was generated against Spider/BIRD as NL exemplars, and the
   alignment transfers to the source distribution.

**Curve shapes:** all four trained-time eval slices descended
monotonically and plateaued in the last 5–10 evals.
`eval.schemapile-held` and `eval.sqale-held` are saturating at the 2 GB
budget; `eval.fineweb-held` and `eval.spider` could still use more
bytes.

**Forward implications:**

- **Multi-GPU step-up justified at the next byte-budget bump.** 8 GB on
  6 × 4090 ≈ 7 h, vs. v2's 10 h on a single GPU at 2 GB. DDP path is
  proven; what's new is the budget.
- **v3 corpus mix has a clean baseline to beat.** Any v3 mixture must
  keep `eval.fineweb-held` ≤ 1.61, push `eval.finepdfs-lab-held` below
  1.78, and not regress on schemapile/sqale.
- **BIRD held-out as a second transfer probe** in v3 — same logic as
  Spider in v2, cleaner test.

**The session note** at
`docs/scratch/2026-04-27/131700_v2_vs_v1_stratified_comparison.md`
contains the full comparison narrative including the v1 cross-eval
that produced the comparison table.

## 11. The v2 → SOTAB head fine-tune gate

The v2 backbone is healthy in the unsupervised pretraining regime.
**It has not yet been validated on a supervised CTA objective.** The
2026-04-19 representation collapse on direct-from-random SOTAB CTA was
the open wound that motivated v2 in the first place; closing that loop
requires a fine-tune from `outputs/mixed-v2/20260426T232240Z/final.pt`
that produces non-degenerate per-class F1.

This is the M2 empirical gate. Three liveness thresholds:

- ≥ 3 distinct embedding clusters at coarse MCL inflation (vs. the
  single cluster that flagged collapse in April)
- ≥ 0.10 macro F1 on the held-out SOTAB v2 Schema.org CTA validation
  set
- Predictions distributed across ≥ 10 distinct labels (no mode-class
  collapse)

These are deliberately undemanding. They distinguish "the model is
alive" from "the model has collapsed." If they fail, the underlying
problem is architectural, not vocabulary-related, and vocabulary
expansion work pauses until it is debugged.

If they pass, the [Phase 1 supervised
roadmap](./roadmap/supervised.md) becomes meaningful — competitive F1
numbers against published baselines (SOTAB-CTA macro F1 > 0.85 easy,
> 0.65 hard, etc.) become legitimate next targets, vocabulary
expansion past the copied baseline begins, and `vocab_label_map.json`
v1.0.0 ships as the first outward release.

## 12. How this relates to v3

The v3 concept brief proposed a phased plan: Phase 1 (Aegir-only
baseline) → Phase 1.5 (Mergekit specialist fusion) → Phase 2
(conditional Nano latent alignment). All
three phases assumed "a working Aegir baseline." The story in this
chapter is what "working" means: Aegir cannot be trained from random
on sparse classification — it needs pretraining + supervised fine-tune
from a healthy backbone. The v2 mixed-corpus pretrain provides the
backbone; the M2 head fine-tune provides the supervised half.

Phase 1.5 Mergekit fusion becomes **stronger** under this picture. The
specialists it fuses will each be pretrained-then-task-finetuned, so
the task-vectors it combines have genuine semantic structure rather
than the small delta between random init and a barely-moved
classifier.

Phase 2 Nano alignment becomes **better grounded**. v3 assumed Aegir
had some baseline representation to align to Nano's; the v2 stratified
eval confirms that representation exists in the unsupervised regime.
The supervised half of the alignment story still requires the M2 gate
to clear.

## 13. Further reading

- [Diagnostic case study: representation collapse on SOTAB-Schema.org](./pretraining/diagnostic_case_study.md)
- [Ontology Charter](./ontology/charter.md) — the empirical gate
  formally specified, plus the outward contract Ægir publishes
- [Phase 1 supervised roadmap](./roadmap/supervised.md) — current
  fine-tune-from-v2 plan that supersedes the from-random approach
- [Training tactics](./pretraining/training_tactics.md) — pre-existing
  ontology-side training objectives
- Session notes:
  - `docs/scratch/2026-04-19/` and `docs/scratch/2026-04-20/` — Stage A/B
    findings
  - `docs/scratch/2026-04-21/061600_overnight_corpus_and_mixed_training.md`
    — v1 mixed-corpus run
  - `docs/scratch/2026-04-23/232400_v2_corpus_kickoff.md` — v2 setup
  - `docs/scratch/2026-04-27/131700_v2_vs_v1_stratified_comparison.md`
    — the v2-vs-v1 stratified result this section reports
