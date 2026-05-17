# Aegir-small SOTAB-CTA v0.1: training-plateau negative finding

**Status:** Negative finding. The lr=1e-4 recovery experiment also plateaued at
random-baseline F1, but along the way it separated two distinct failure modes
that the lr=5e-4 run had conflated.
**Date:** 2026-05-17
**Runs:**
- `20260517T024411Z_c96ba81257_small_sotab` — lr=5e-4, 12 epochs requested, killed after 2.
- `20260517T031437Z_c96ba81257_small_sotab` — lr=1e-4, 3 epochs.

**Outcome:** Did **not** push the Aegir v0.1 checkpoint to HuggingFace. Dataset
and LoRA artifacts are already published; the Aegir model card stays in-repo
as a scaffold pending an architectural fix.

## What was attempted

Aegir-small (`d_model=[256,384,384]`, `arch_layout=["w4",["w4",["w8"],"w4"],"w4"]`,
~56M params) trained end-to-end on SOTAB v2 Schema.org CTA. Effective batch
size 96 (16 × 6 GPUs), bf16 AMP, cosine LR with linear warmup, λ_lb=0.1,
target downsample factor 2.0 per stage. Two runs:

- **run A** (lr=5e-4, 12 epochs requested) — killed after 2 epochs.
- **run B** (lr=1e-4, 3 epochs) — completed.

## What happened

Both runs plateaued at train loss ~4.10 (random baseline `ln(82) = 4.41`).
Val loss > random in both cases; val F1 macro ≈ 0 in both cases. But the
two runs **diverged in their failure mode at the chunker level** — and that
difference is the most informative thing we got.

### Headline metrics

| Run | Epoch | train_loss | task_loss | val_loss | micro F1 | macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| A — lr=5e-4 | 1 | 4.1250 | 4.0233 | 4.6208 | 0.0305 | 0.00076 |
| A — lr=5e-4 | 2 | 4.1091 | 4.0094 | 4.6087 | 0.0102 | 0.00026 |
| B — lr=1e-4 | 1 | 4.1112 | 4.0100 | 4.6053 | 0.0203 | 0.00051 |
| B — lr=1e-4 | 2 | 4.0967 | 3.9980 | 4.6002 | 0.0136 | 0.00034 |
| B — lr=1e-4 | 3 | 4.0899 | 3.9916 | 4.5973 | 0.0136 | 0.00034 |

`val_loss > ln(82) ≈ 4.41` in every epoch — the model is systematically worse
than uniform on val, not just under-trained. Train F1 ~0.04 micro / ~0.004
macro across both runs and never moves materially.

Lowering LR by 5× moved train loss by ~0.014 nat and made val F1 *worse*
than at lr=5e-4. **Best val macro F1 of the entire 3-epoch lr=1e-4 run was
at epoch 1 (0.00051) — then it regressed and never recovered.** Train loss
keeps slowly drifting down (4.111 → 4.097 → 4.090) while val loss / F1
oscillate. The model is overfitting to a noisy or mis-aligned signal,
not learning the task.

Hyperparameter tuning is not the lever here.

### Boundary diagnostics — two failure modes, separated

| Run | Epoch | stage0 mean_F | stage0 achieved_N | stage0 abs_err | stage1 mean_F | stage1 achieved_N | stage1 abs_err |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — lr=5e-4 | 1 | 0.352 | 2.89 | 0.148 | 0.576 | 1.80 | 0.099 |
| A — lr=5e-4 | 2 | 0.653 | 1.54 | 0.153 | **0.047** | **24.82** | **0.453** |
| **B — lr=1e-4** | 1 | 0.524 | 1.92 | 0.040 | 0.579 | 1.75 | 0.089 |
| **B — lr=1e-4** | 2 | 0.532 | 1.89 | 0.045 | 0.552 | 1.84 | 0.068 |
| **B — lr=1e-4** | 3 | 0.537 | 1.87 | 0.048 | 0.562 | 1.80 | 0.079 |

At lr=5e-4, the deepest-stage chunker collapsed between epoch 1 and 2:
stage1 mean_F dropped 0.576 → 0.047 and achieved_N exploded 1.80 → 24.82.
With max_length=128, stage0 keeping ~half the tokens, and stage1 then
flagging only ~5% of those as boundaries, the deepest hierarchical level
ended up operating on ~3 tokens per sequence — information flow through
the recursion collapsed.

**At lr=1e-4 the chunker stayed healthy through all three epochs.** Both
stages stayed near their target downsample factor (achieved_N ≈ 1.87–1.92
for stage0, 1.75–1.84 for stage1; abs_err < 0.10 throughout). The chunker
collapse is real, it is LR-sensitive, and lr=1e-4 fully prevents it.

**But preventing the collapse did not unblock learning.** Even with a
stable chunker and tight boundary statistics, val F1 macro stayed at 0
and val loss stayed above random baseline. So the plateau is **not**
caused by chunker collapse — collapse is a downstream consequence of
high LR, but the plateau is its own, separate failure.

## Why — two-failure-mode interpretation

1. **Chunker collapse (LR-sensitive).** At lr=5e-4 the STE-binarized
   boundary indicators settle into a degenerate equilibrium where the
   deepest stage is effectively bypassed — task loss is locally stationary,
   the chunker drifts under noise rather than task signal, and the LB
   loss (a smooth function of the *marginal* boundary mean, weighted at
   λ=0.1) is too weak to prevent it. At lr=1e-4 the chunker updates are
   small enough that it stays in target territory.

2. **Task plateau (LR-independent).** Even with a healthy chunker, the
   model does not learn the SOTAB-CTA classification signal. Train task
   loss = 3.998 at lr=1e-4 epoch 2 — ~0.4 nat below random baseline,
   which is "tiny but nonzero." Val F1 macro = 0. The model is finding
   *some* signal on train, but it isn't generalizing — and the train
   signal itself is small enough that this looks more like overfitting
   to noisy data-augmentation than learning structure.

## What this rules in / out

Ruled **out**:
- LR magnitude as the cause of the task plateau.
- Chunker collapse as the cause of the task plateau (independent dimension).
- DDP unwrap bug (fixed in `ae8b829`).
- Missing step-level logging (fixed in `c96ba81`).
- Random-init wreckage (warmup steps clearly move the loss downward
  before the asymptote).

Ruled **in**:
- High LR causes chunker collapse via the STE/LB-loss interaction.
  Lr=1e-4 fixes that specifically.
- The remaining task plateau is **architectural and/or data-pipeline**.
  Same scaffold has trained successfully on synthetic data
  (`train.py --smoke-test`), so the failure is specific to the
  SOTAB-CTA real-data path.

## What to investigate (post-v0.1)

The two failure modes need separate fixes.

### Task plateau (the harder, more important problem)

The plateau is the blocker. Even with stable chunking, val F1 macro is
zero. Candidate causes, ordered roughly by likelihood:

- **Data-pipeline bug.** Same architecture trains on synthetic data; it
  fails on SOTAB-CTA. Sanity-check what the model actually sees:
  serialized byte sequences for a known table, what labels are mapped
  to, whether the label index maps correctly through `TASK_NUM_CLASSES`
  (recently changed from 91 → 82 — verify the train/val CSV unions
  match this).
- **Loss weighting.** Class-frequency imbalance in SOTAB-CTA is severe
  (top class is `Product`, tail classes have < 100 samples). The
  current `CrossEntropyLoss` uses uniform weighting; macro F1 = 0
  suggests the model collapses to predicting the majority class on
  val (which barely shows up in macro F1 even with reasonable accuracy
  on `Product`).
- **Context too short.** `max_length=128` byte-level is ~15 ASCII
  chars after MMR-selected 8 context cols. That may simply not contain
  enough signal. Try `max_length=512` first and see if val F1 moves.
- **Head architecture.** `AegirForColumnAnnotation` reads from the
  byte-level output after the full hierarchical roundtrip; if the
  dechunk is information-degrading even when chunker stats look
  healthy, the head sees impoverished features. Consider tapping
  the deepest-stage representation instead of after the full dechunk.

### Chunker collapse (LR-sensitive, has a known workaround)

- **lr ≤ 1e-4 keeps the chunker stable** in this regime — the
  immediate prescription is to ship the next attempt at that LR.
- **LB-loss formulation.** The marginal-mean penalty is too weak when
  STE-binarized boundaries collapse to a delta. Worth replacing with
  something that constrains the *local* distribution (entropy
  regularizer on boundary scores, or KL to a target Bernoulli).
- **Decouple chunker / backbone LRs.** Routing module is small;
  may need a different LR than the rest.
- **Warm-start.** Start with `lambda_lb` high and decay it; or
  warm-start the routing module with a brief Bernoulli-target
  supervised phase.

### General hygiene

- Smaller models first. Validate end-to-end on tiny
  (`d_model=[128,192,192]`) for many epochs before scaling. If tiny
  also plateaus on SOTAB-CTA real data, the issue is in the data
  pipeline / loss / head, not model capacity.
- Add a "data sanity" diagnostic to `train.py`: print the first
  few training (input bytes, label) pairs at epoch 0, so we'd
  catch a label-mapping bug in the first 10s of any run.

## Decision

- **Do not push** `zndx/aegir-sotab-cta-v0.1` while it's at random-
  baseline F1. Card scaffold + eval driver + push script + diagnostic
  hooks all live in-repo under `data/aegir-sotab-cta-v0.1/` and
  `scripts/aegir_*` — ready for the next attempt with no rework.
- **Already shipped tonight, healthy:**
  - `zndx/sdg-bertopic-correspondence-v0.1` (dataset, peer-review-ready)
  - `zndx/sdg-sft-r1` (LoRA)
  - `zndx/sdg-sft-r2` (LoRA)
- **Next-attempt prescription:** at minimum, ship with `--lr 1e-4`
  (chunker-safe). Before scaling, investigate the data-pipeline /
  loss-weighting hypotheses, because LR alone clearly is not the
  primary issue.
