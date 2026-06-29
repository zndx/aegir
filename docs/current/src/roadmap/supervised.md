# Phase 1: Supervised Bootstrapping

Phase 1 fine-tunes a CTA/CPA head from the v2 mixed-corpus pretrain
checkpoint, *not* from random initialization. The original Phase 1
plan (train from random on Column Type Annotation) was invalidated by
the 2026-04-19 representation-collapse incident on SOTAB v2
Schema.org CTA; the v2 byte-level pretrain (2026-04-27) produces the
well-conditioned starting point that the fine-tune proceeds from. The
fine-tune is the M2 empirical gate.

Train the Aegir column-annotation head on Column Type Annotation (CTA)
and Column Property Annotation (CPA) benchmarks, **starting from the
v2 byte-level pretrain checkpoint at
`outputs/mixed-v2/20260426T232240Z/final.pt`**. This phase establishes
baseline performance and demonstrates that the pretrained backbone
escapes the failure mode that the from-random approach hit.

## Objective

Produce a single Aegir checkpoint that achieves competitive F1 scores
on standard CTA/CPA benchmarks, operating directly on raw byte
sequences (no external tokenizer), via head fine-tune from a pretrained
backbone.

## Why we are not training from random

The 2026-04-19 SOTAB-CTA run (small config, 56M params, 3 epochs, lr
3e-4) produced complete representation collapse:

- Every val sample produced the identical pooled embedding to within
  bf16 rounding noise (max pairwise L2 = 0.020 on vectors of mean norm
  6.98).
- The classifier predicted `currency` on 100% of 1500 val samples;
  exact-match accuracy equalled the val base rate of the mode class.
- MCL geometry audit returned one cluster at every inflation tested.

A subsequent hygiene-only rerun (lr 5e-5, weight decay 1e-4, warmup,
gradient clipping) reproduced the collapse to within 1 part in 10³ on
both train and val loss. This is not a hyperparameter bug. The
underlying issue is that H-Net + RWKV-7 is architecturally a language
model — its mechanisms (dynamic chunker boundary learning, RWKV-7 time
decay, recurrent state evolution) are designed for dense per-token
supervision, not for sparse classification gradients. From-random
direct CTA does not give the architecture the gradient signal it needs
to stabilize.

The full diagnosis is in
[Diagnostic Case Study](../pretraining/diagnostic_case_study.md) and
the staged plan that resolved it is in
[Training Regime](../training_regime.md).

## What changed since the original plan

| Element | Original plan | Current plan |
|---|---|---|
| Starting point | Random initialization | v2 mixed-corpus pretrain checkpoint |
| First objective | Direct CTA softmax over 91 SOTAB labels | Head fine-tune; backbone optionally frozen for the first probe |
| Success criterion | F1 > 0.85 macro on SOTAB-CTA easy | Liveness gate first (≥ 0.10 macro F1, ≥ 3 MCL clusters, ≥ 10 distinct predicted labels). Competitive F1 numbers are downstream of liveness. |
| Vocabulary | Schema.org 91 labels (stale; correct count is 82) | 82 labels via `vocab_label_map.json`; multi-benchmark support via `_LABEL_DIMS` keys |
| Loss design | Categorical cross-entropy on leaf labels | Same for the first probe; hierarchical path-prediction is a Stage C extension once liveness is established |
| Compute | Up to 6 × RTX 4090 DDP at small config | Single GPU sufficient for fine-tune from healthy v2 backbone; multi-GPU is M3 not M2 |

## Target Datasets

| Dataset | Task | Tables | Columns | Label Classes |
|---------|------|--------|---------|---------------|
| SOTAB-CTA (Schema.org) | Column Type Annotation | ~50k | ~500k | 82 (verified — fixes stale 91) |
| SOTAB-CTA (DBpedia) | Column Type Annotation | ~50k | ~500k | 101 / 53 (full / restricted) |
| GitTables | CTA (large-scale) | ~1.5M | ~15M | 122 DBpedia types |
| WikiTables | CTA/CPA | ~1.7M | ~6M | DBpedia ontology |

## Liveness gate before competitive F1 targets

Per the [v2 → SOTAB head fine-tune gate](../training_regime.md#11-the-v2-sotab-head-fine-tune-gate),
the v2 → SOTAB head fine-tune must clear three liveness checks before
any "competitive F1" target is meaningful:

- ≥ 3 distinct embedding clusters at coarse MCL inflation
- ≥ 0.10 macro F1 on the SOTAB v2 Schema.org CTA validation set
- Predictions distributed across ≥ 10 distinct labels (no mode-class
  collapse)

These are deliberately undemanding. They distinguish "the model is
making different predictions for different inputs" from "the model has
collapsed to a constant function." Until they pass, the F1 targets
below are aspirational; once they pass, they become the next thing to
optimize.

## Aspirational F1 targets (post-liveness)

| Benchmark | Metric | Target |
|-----------|--------|--------|
| SOTAB-CTA (easy) | Macro F1 | > 0.85 |
| SOTAB-CTA (hard) | Macro F1 | > 0.65 |
| SOTAB-CPA | Macro F1 | > 0.75 |

Source: published REVEAL and SOTAB baselines. These are competitive,
not state-of-the-art; SOTA on SOTAB-CTA is in the high 0.8s for
specialized fine-tunes. We pursue them only after liveness is
established.

## Byte-Level Input

Aegir operates on raw byte sequences (`vocab_size=65536` to cover byte
values plus special tokens). Tables are serialized into a linear byte
stream with role markers distinguishing the target column from context
columns.

**Dynamic chunking learns tokenization from raw bytes.** The
`RoutingModule` in the hierarchical backbone predicts chunk boundaries
based on cosine similarity between adjacent hidden states. The v2
pretrain has already given the chunker a healthy boundary distribution
on natural-language and DDL-flavored bytes; the fine-tune extends it to
table serializations without re-learning byte statistics.

## Serialization Format

Tables are serialized using the format in
`src/aegir/data/serialization.py`:

```
[CLS] col_name: val1 | val2 | val3 [SEP] ctx_col1: v1 | v2 [SEP] ctx_col2: ...
```

The target column comes first, followed by context columns selected
via MMR (Maximal Marginal Relevance) to maximize diversity while
staying within the byte budget.

## Training Configuration

### Fine-tune from v2 (recommended path)

```bash
uv run --no-sync python train.py \
    --task sotab-cta \
    --model-size small \
    --pretrained outputs/mixed-v2/20260426T232240Z/final.pt \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --warmup-steps 500
```

Hygiene parameters (lr 1e-4 with warmup, gradient clipping at
`max_norm=1.0`, weight decay 1e-4) are the same as the v2 pretrain. The
pretrained backbone is in a well-conditioned region of parameter
space; the fine-tune does not need to escape from a saturating decay
basin.

### Single-GPU sufficient for the liveness gate

The liveness gate does not require multi-GPU. A single 4090 fine-tunes
the small-config backbone on SOTAB-CTA in under an hour at the budgets
that matter for liveness. Multi-GPU step-up belongs to M3, after the
gate clears and we are pushing for the aspirational F1 targets.

### Model Sizes (current, verified)

| Size | `d_model` | `arch_layout` | Approx params |
|------|-----------|---------------|---------------|
| tiny | `[128, 192, 192]` | `["w2", ["w2", ["w4"], "w2"], "w2"]` | ~13.5M |
| small | `[256, 384, 384]` | `["w4", ["w4", ["w8"], "w4"], "w4"]` | ~56M (Apr 19 SOTAB run) |
| base | `[768, 1024, 1024]` | `["w4", ["w4", ["w12"], "w4"], "w4"]` | target ~500M |

The Apr 19 representation-collapse run was at `small`; the v2
pretrained backbone matching it is the natural starting point for the
liveness gate. `base` is the target for competitive F1 numbers post-M2.

## Success Criteria

Phase 1 is complete when, in order:

1. The v2 → SOTAB head fine-tune passes the liveness gate
   (`docs/current/ontology/charter.md`).
2. Dynamic chunking continues to produce stable boundary predictions on
   the table-byte distribution (no degenerate all-boundary or
   no-boundary patterns under the fine-tune).
3. The model meets or exceeds aspirational F1 targets on
   SOTAB-CTA/CPA at `base` config.
4. The trained checkpoint is frozen and used as a specialist in the
   far-future Phase 3 PARL training.
