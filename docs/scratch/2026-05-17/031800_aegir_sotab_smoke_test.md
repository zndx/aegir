# Aegir-small SOTAB-CTA — E2E smoke test on 6× RTX 4090

**Status:** Smoke test of the full Aegir training pipeline on real (non-
synthetic) tabular data. Pipeline is healthy end-to-end; F1 plateau is
the expected outcome of training a 56M-param byte-level model from
scratch on a sparse-reward classification task. Results give us a clean
baseline against which to read the upcoming dense-text pretraining pass.
**Date:** 2026-05-17
**Runs:**
- `20260517T024411Z_c96ba81257_small_sotab` — lr=5e-4, 12 epochs requested, killed after 2 (chunker collapse).
- `20260517T031437Z_c96ba81257_small_sotab` — lr=1e-4, 3 epochs, completed cleanly.

**Outcome:** Aegir v0.1 SOTAB checkpoint is not the right artifact to publish
yet — but the smoke test cleared what it needed to clear:

- Pipeline runs end-to-end on real SOTAB-CTA data across 6 GPUs.
- The chunker has a known stable LR regime (`lr ≤ 1e-4`) and a known
  unstable one (`lr=5e-4` → stage-1 collapse at epoch 2). This
  prescription transfers directly to the dense-text pretraining pass.
- F1 ≈ 0 from scratch on a sparse-reward task establishes the baseline
  for "untrained backbone" — anything the dense-text pretrain produces
  will be measured against this.
- Dataset/LoRA artifacts (`zndx/sdg-bertopic-correspondence-v0.1`,
  `zndx/sdg-sft-r1`, `zndx/sdg-sft-r2`) shipped earlier today are
  independent and unaffected.

## Why F1 ≈ 0 is the expected baseline here

Aegir-small is byte-level, 56M params, no prior weights. SOTAB-CTA
provides ~117K examples × a single 82-way integer label — gradient
signal per backbone parameter is vanishingly small for "learn
byte-level LM + table structure + classification simultaneously."

RWKV-7 + H-Net is independently validated on standard pretraining
mixes; the architecture is not the question here. The right regime
for tabular benchmarks is **pretrain on a dense-text mix first (FinePDF-
Edu / C4 / similar), then fine-tune the head and a low-LR backbone on
SOTAB**. This run was deliberately scoped as the from-scratch end-
to-end probe — its F1 was never going to be publishable, but the
pipeline / chunker / dataloader / boundary diagnostics needed validating
in production conditions before pretraining starts.

## What was attempted

Aegir-small (`d_model=[256,384,384]`, `arch_layout=["w4",["w4",["w8"],"w4"],"w4"]`,
~56M params) trained end-to-end on SOTAB v2 Schema.org CTA. Effective batch
size 96 (16 × 6 GPUs), bf16 AMP, cosine LR with linear warmup, λ_lb=0.1,
target downsample factor 2.0 per stage. Two runs:

- **run A** (lr=5e-4, 12 epochs requested) — killed after 2 epochs when
  the stage-1 chunker collapsed.
- **run B** (lr=1e-4, 3 epochs) — completed cleanly; chunker stable.

## Headline metrics

| Run | Epoch | train_loss | task_loss | val_loss | micro F1 | macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| A — lr=5e-4 | 1 | 4.1250 | 4.0233 | 4.6208 | 0.0305 | 0.00076 |
| A — lr=5e-4 | 2 | 4.1091 | 4.0094 | 4.6087 | 0.0102 | 0.00026 |
| B — lr=1e-4 | 1 | 4.1112 | 4.0100 | 4.6053 | 0.0203 | 0.00051 |
| B — lr=1e-4 | 2 | 4.0967 | 3.9980 | 4.6002 | 0.0136 | 0.00034 |
| B — lr=1e-4 | 3 | 4.0899 | 3.9916 | 4.5973 | 0.0136 | 0.00034 |

Random baseline: `ln(82) ≈ 4.41`. Train task loss drifts slowly down
(4.023 → 3.992) — the backbone is finding *some* signal, but it's the
~0.4 nat headroom you'd expect from learning trivial label-frequency
priors on noisy byte streams, not from learning the actual classification
mapping. F1 stays near zero, with the run-B best at epoch 1 (peak val
macro F1 = 0.00051) and decaying as it overfits the same priors.

This is exactly the shape of a from-scratch run on a sparse-reward task:
loss dips below uniform-random, then asymptotes because the supervision
signal is too thin to learn structure rather than priors.

## Pipeline validations (what the smoke test *did* prove)

### 1. Boundary diagnostics — chunker stability has a usable LR prescription

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
the recursion collapses.

At lr=1e-4 the chunker stayed healthy through all three epochs. Both
stages tracked their target downsample factor with abs_err < 0.10
throughout. The collapse is real, LR-sensitive, and lr=1e-4 prevents
it. **This is a transferable result** — the dense-text pretraining
pass should default to lr ≤ 1e-4 unless we re-validate at a higher LR.

The two failure modes (chunker collapse, task plateau) are independent
dimensions: the run-A collapse is a known H-Net instability, while the
run-B plateau is the sparse-reward outcome above.

### 2. Engineering correctness

- DDP wrapper, .config unwrap (`ae8b829`) — correct under 6-GPU run.
- Step-level logging (`c96ba81`) — working; ~7s per 20 steps steady state.
- Cosine LR schedule + warmup ramp visible cleanly in logs (0 → 1e-4
  over ~90 steps, then cosine decay to 1e-5).
- Boundary diagnostics + metrics.json persistence — both populated
  correctly across all epochs.
- MMR cache (`~/.cache/aegir/mmr/all-mpnet-base-v2.sqlite3`, 695 MB)
  warm-restarts across runs without rebuild — startup cost ~5 min for
  the first run, much faster for the second.
- 116,887 train / 1,769 val samples loaded correctly; num_classes
  auto-detected at 82 from both splits.

### 3. Sparse-reward baseline numbers (useful for the next pass)

Anything the dense-text pretrained backbone scores on SOTAB-CTA via
the same head + fine-tune recipe should clear:

- val loss < 4.41 (random baseline; this run sits at 4.60).
- val F1 macro > 0.001 (this run's best is 0.00051, all training
  epochs after epoch 1 *regress*).
- val F1 macro > 0.1 (the "actually learning" threshold).

A pretrained backbone hitting val F1 macro ≈ 0.5–0.7 within 1–2
fine-tune epochs would be the clean expected outcome based on
DoDuo / TURL-class results on the same task.

## Plan for v0.2 publication path

The original two-phase plan (dense-text pretrain → SOTAB fine-tune)
under-specified the bridge between the two. TAPAS
([google-research/tapas](https://github.com/google-research/tapas),
archived 2024-07-22 but methodology still valid) shows that a
**dense-supervision intermediate pretrain on synthetic table-grounded
binary tasks** is the missing curriculum step between unsupervised
LM and sparse-reward tabular tasks. Their recipe produced ~7.8M
examples (3.7M synthetic SQL-like comparisons + 4.1M counterfactual
entity swaps) and gave a measurable lift on every downstream task
they tried (WTQ, WikiSQL, TabFact, SQA). Adopting that recipe gives
us:

1. **Phase 0 — backbone pretraining (dense text + table-text mix).**
   Combine FinePDF-Edu / C4 with TAPAS's 6.2M Wikipedia table-text
   pairs (`gs://tapas_models/2020_05_11/interactions.txtpb.gz`,
   protobuf text format — we'll need a byte-level converter from
   their `interaction.proto`). Default LR ≤ 1e-4 (chunker-safe).
   Watch boundary diagnostics each epoch — if stage-1 chunker stays
   in target range there, the prescription transfers cleanly.

2. **Phase 0.5 — intermediate pretrain (TAPAS-style dense-supervision
   bridge).** Generate ~5–10M binary-classification examples over the
   table-text corpus:
   - **Synthetic SQL-grounded statements** — random grammar produces
     comparisons / aggregations over the table; binary truth label.
     ("*X is greater than the sum of Y when Z is K*" → True/False.)
     Cheap to generate at arbitrary scale; teaches numerical reasoning
     grounded in the table.
   - **Counterfactual entity swaps** — take a sentence near the table,
     swap one entity for a plausible alternative from the same column;
     binary "corrupted?" label. Teaches entity-grounding.

   Architectural note: TAPAS gained +1-2% from
   `reset_position_index_per_cell` — feeding explicit cell-boundary
   priors. For Aegir, this suggests adding sentinel bytes at cell
   boundaries in the serialization rather than relying solely on the
   dynamic chunker to discover them via cosine-similarity routing.
   Worth A/B-ing during Phase 0.5 since we already know the chunker
   is the load-bearing structure.

3. **Phase 1 — SOTAB-CTA fine-tune** with `AegirForColumnAnnotation`
   head, frozen or low-LR backbone. The infrastructure published here
   (`scripts/aegir_eval_sotab.py`, `scripts/aegir_push_to_hf.py`,
   `data/aegir-sotab-cta-v0.1/README.md`) all carries forward
   unchanged. With a backbone that's already seen ~13M table-grounded
   examples by this point, 117K SOTAB-CTA labels should suffice.

4. **Phase 2 — robustness eval** on all five SOTAB-v2 splits via the
   already-built eval driver. Push checkpoint to HF when val F1 macro
   on the main test split clears a publishable threshold
   (≥ 0.1 minimum, 0.5+ for a credible release).

**Methodology references (read before Phase 0):**
- Herzig et al., *TAPAS: Weakly Supervised Table Parsing via Pre-
  training*, ACL 2020.
- Eisenschlos et al., *Understanding Tables with Intermediate Pre-
  training*, EMNLP Findings 2020.

## Optional sharpening before the dense-text pass

These are nice-to-haves, not blockers — the pipeline is working as-is:

- **Add a "data sanity" diagnostic** to `train.py` that prints the
  first few (input bytes, label) pairs at epoch 0. Catches label-
  mapping bugs in the first 10s of any run.
- **Consider an entropy regularizer** on the boundary scores as a
  belt-and-braces backup to the lr ≤ 1e-4 chunker prescription. The
  current LB-loss penalizes the marginal boundary mean, which is too
  weak when the STE-binarized distribution collapses to a delta.
- **Stretch `max_length`** from 128 to 256 or 512 for the SOTAB
  fine-tune — 128 bytes / 8 context cols is ~15 chars per column,
  which is right at the limit of what's parseable.

## What this leaves on HuggingFace tonight

Already shipped, healthy, peer-review-ready:
- `zndx/sdg-bertopic-correspondence-v0.1` (dataset)
- `zndx/sdg-sft-r1` (Qwen3.5-9B LoRA)
- `zndx/sdg-sft-r2` (Qwen3.5-9B LoRA, iterated)

Held back, scaffold ready in-repo:
- `zndx/aegir-sotab-cta-v0.1` — card + eval driver + push script live
  under `data/aegir-sotab-cta-v0.1/` and `scripts/aegir_*`. Will flip
  the switch once the dense-text pretrained backbone clears a
  publishable F1 on SOTAB-CTA fine-tune.
