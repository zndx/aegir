# SOTAB diagnostic — representation collapse, not "weak learning"

The upgraded diagnostic ran on the SOTAB-small checkpoint (3 epochs,
`outputs/best_model.pt`, run `20260419T182156Z_d8feb9905f_small_sotab`).
Findings are decisive and change what "fix the plateau" means.

## Headline

**The model didn't fail to learn — it actively collapsed its
representation space.** Every val sample produces the *identical*
pooled embedding to within bf16 rounding noise.

- Max pairwise L2 across 50 random val samples: **0.020**
- Mean embedding norm: **6.98**
- Ratio: **2.9e-3** (threshold for "collapsed" in our diagnostic: 1e-2)
- Per-dim variance across 1500 samples: **max 2.96e-06, median 4.51e-08**
- Predictions: **100% `currency`** (1500 / 1500 val samples)
- Exact-match acc: **3.27%** — exactly the base rate of `currency` in val
- MCL at every inflation: **1 cluster** (everything is one point in
  embedding space)

## What this means vs what I expected

I went into the diagnostic expecting to distinguish between two
hypotheses:

1. Features cluster by label, classifier head isn't using them → loss
   function fix (dual center loss, hierarchical softmax, etc.)
2. Features are diffuse and meaningless → upstream fix (context,
   serialization, architecture).

Neither hypothesis fits. The features aren't diffuse and they aren't
clustered; they are **mathematically a single point**. The classifier
head sees the same input on every forward pass, so its argmax defaults
to the same class regardless of what column it "saw."

This ALSO means MCL-based geometry audit can't even begin to answer
the original question ("does the embedding space have hierarchical
structure?"). You need a representation that varies with input before
you can ask whether the variation is hierarchical. The audit is
postponed until we have a model that isn't collapsed.

## Where the collapse lives

Post-run state inspection of `best_model.pt`:

| Layer | Shape | abs-mean | norm |
|-------|-------|----------|------|
| pooler.weight | (256, 256) | 2.89e-2 | 8.68 |
| pooler.bias | (256,) | 3.31e-2 | 0.61 |
| classifier.weight | (91, 256) | 3.33e-2 | 6.42 |
| classifier.bias | (91,) | 3.81e-2 | 0.44 |
| residual_proj.weight | (256, 256) | 6.08e-3 | 2.53 |
| residual_proj.weight (inner) | (384, 384) | 2.26e-3 | 1.33 |

All heads and the residual projection are healthy — non-dead weights
with reasonable magnitudes. So the collapse is **upstream of the
heads, inside the Aegir backbone itself** (RWKV time-mix / H-Net
dynamic chunking / DeChunk EMA).

Boundary diagnostics recorded per-epoch during training do NOT show
collapse: `stage0_mean_F` stayed in 0.51–0.56, `stage1_mean_F` in
0.33–0.65 across all 3 epochs. The dynamic chunking pipeline itself
was doing healthy-looking work.

The suspicious layers that remain:

- **RWKV-7 time-decay parameter** (`w`). Known to destabilize at high
  learning rates — if `w` saturates such that the recurrent state is
  either reset-every-step or never-updated, every layer's output
  becomes position-independent. Combined with H-Net pooling at
  cls_indexes, this produces a constant output.
- **Value-first sharing** (`v_first`). Layer 0 sets it, later layers
  lerp via a LoRA gate. If layer 0's v_first is input-independent
  (e.g., dominated by its bias under a saturated gate), every
  downstream layer inherits an input-independent signal.
- **STE residual composition**. STE.forward returns ones, so at init
  the model passes through the inner path at full strength. If the
  inner path collapses (via the above), the residual_proj (zero-init
  weight, small bias) can't rescue it.

I did **not** instrument which of these three is the specific failure
mode — that's a follow-on investigation, one layer at a time. The
immediate action doesn't depend on knowing which: the fix is
training-regime-level, not architecture-level.

## Why gt-signals didn't collapse, SOTAB did

|  | gt-signals-dbpedia | SOTAB-Schemaorg-CTA |
|---|---|---|
| Train samples | 1,999 | 116,887 |
| Epochs | 20 | 3 |
| Total gradient steps | ~2,500 | ~22,000 |
| Best val macro F1 | **0.126** | **0.0007** |
| Embedding collapse | no (F1 rose meaningfully) | **yes** |

**10× more gradient steps at the same learning rate + same model**
is the differentiator. My working hypothesis: at `lr=3e-4` with no
grad clip and only ~100 steps of warmup, SOTAB's ~22k updates push
the RWKV parameters into a basin where the recurrence degenerates.
gt-signals' ~2.5k updates don't get that far.

Known-quantity support: the RWKV-LM training recipes in the
`ref/rwkv-lm/` repo use **much** smaller learning rates for
from-scratch runs (typically 1e-4 with warmup 1000+ steps) and
**always** use grad clip. We have neither.

## Recommended fix (one training-regime intervention, not four)

Re-run SOTAB with a single bundled change set:

1. **Learning rate**: `3e-4 → 5e-5` (6× reduction, matches RWKV
   from-scratch norms).
2. **Warmup**: explicit 1000-step warmup (currently whatever the
   default is from our train.py; probably too short for 22k updates).
3. **Gradient clipping**: `max_norm=1.0`. This is a one-line addition
   to the training loop that's absent today.
4. **Weight decay**: `1e-2 → 1e-4`. AdamW's default is aggressive for
   recurrent architectures.

These are all the same "RWKV training hygiene" intervention — each
supports the others. Isolating each is a second-order ablation.

## Secondary finding

The 91 SOTAB Schema.org labels roll up to a **very long-tail parent
distribution**: 85% of val labels roll up to a single Schema.org
top-level parent. We discovered this incidentally because MCL's
one-cluster output still had 85% purity against the parent label.
That's a property of the task, not the model — but it means when we
do get a working representation, the dual-center-loss hierarchical
anchor will have a naturally-dominant parent cluster. Worth
remembering when we design the actual loss.

## Action

1. Commit the diagnostic script + artifacts.
2. Re-train SOTAB with the bundled hygiene fix (proposed above).
3. Re-run the diagnostic on the new checkpoint. If collapse resolves,
   the MCL inflation sweep becomes informative and we can answer the
   original question about hierarchical geometry.
4. Only THEN evaluate dual-center-loss as a loss-function improvement.

The dual-center-loss discussion was the right architectural
conclusion for this *task*, but it's not the right *next action* —
the model has to actually learn something first.
