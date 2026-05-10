# Diagnostic Case Study: Representation Collapse on SOTAB

*A short postmortem of the first SOTAB training attempt. The detailed
technical note lives in `docs/scratch/2026-04-19/234700_sotab_diagnostic_representation_collapse.md`;
this chapter extracts the reusable methodology and the lessons.*

## 1. The signal

After 3 epochs of direct SOTAB CTA training, the loss and F1 curves
looked like a plateau: train loss 4.13 → 4.10, val loss 4.55 → 4.52,
best val macro F1 = 0.0007 reached at epoch 1 and never improved. A
casual interpretation is "the model is having trouble learning a hard
task." That interpretation is wrong.

## 2. The three-phase diagnostic

The script at `scripts/sotab_diagnostic.py` runs three orthogonal
analyses on any trained checkpoint. Each answers a different question;
reading them together pinpoints the failure mode.

### Phase 1 — Prediction distribution

Count what the classifier is actually predicting across the val set.
Report the top-k predicted classes and the exact-match accuracy.

**Signal read**:

- If a single class accounts for ≥50% of predictions → *mode collapse*.
- If predictions are evenly distributed but wrong → *learning rate or
  schedule problem*.
- If predictions are concentrated in a plausible few classes but mixed
  up → *confusable-class problem*, needs per-class analysis.

On our collapsed SOTAB checkpoint: 100% predictions of `currency`,
exact-match 3.27% equalling the val base rate of that class.

### Phase 2 — Cluster geometry

Extract pre-classifier pooled embeddings. Compute:

- Mean embedding norm (scale of representation)
- Per-dimension variance across samples (spread)
- Max pairwise L2 distance, probed on 50 random sample pairs
- Intra-class vs inter-class cosine distance on normalized embeddings

**Collapse detector** (the most important signal): the ratio of max
pairwise L2 to mean embedding norm. If below 1% (equivalently, the
spread across samples is below rounding noise on the mean vector),
the representation has collapsed to a single point. All downstream
cluster analyses become mathematically degenerate.

On our collapsed checkpoint: max pairwise L2 / mean norm = 2.9 × 10⁻³,
well below the 10⁻² collapse threshold. Per-dimension variance max
2.96 × 10⁻⁶, median 4.51 × 10⁻⁸.

### Phase 3 — MCL inflation sweep (van Dongen)

Build the cosine similarity graph of the embeddings. Run MCL at
inflation ∈ {1.4, 2.0, 3.0, 4.0}. Report cluster count and cluster
purity against both leaf labels and a parent-bucket mapping (for
Schema.org, the top-level types: Organization, Place, CreativeWork,
Intangible, etc.).

**Signal read**:

- Cluster count decreases monotonically as inflation decreases.
- Parent-level purity *rises at coarser inflations* → embedding
  geometry encodes the ontology hierarchy. The classifier head is
  the bottleneck; the representation is healthy.
- Parent purity is *flat across inflations* → no hierarchical
  structure in the embeddings. Either the representation itself is
  weak, or the task-ontology pairing isn't well-encoded. Loss
  function fixes won't help.
- *One cluster at every inflation* → representation collapse, audit is
  suspended pending a non-collapsed checkpoint.

On our collapsed checkpoint: one cluster at every inflation. The
reported parent purity of 0.851 is simply the fraction of val labels
whose Schema.org parent is the *modal* parent — a property of the
label distribution, not the embedding geometry.

## 3. Localising the collapse

Weight inspection of the saved checkpoint:

| Layer | Norm | Status |
|---|---|---|
| pooler.weight (256×256) | 8.68 | healthy |
| pooler.bias (256) | 0.61 | healthy |
| classifier.weight (91×256) | 6.42 | healthy |
| classifier.bias (91) | 0.44 | healthy |
| residual_proj.weight, outer (256×256) | 2.53 | healthy |
| residual_proj.weight, inner (384×384) | 1.33 | healthy |

No dead heads. The collapse is upstream, inside the backbone.

Boundary-predictor diagnostics recorded per-epoch during training stay
in the healthy 0.33-0.65 range (mean_F at each stage), so the dynamic
chunker is NOT the mechanism. Suspects in the backbone remain:

1. **RWKV-7 time-decay saturation** — `w` parameter drifting to a
   fixed point where the recurrent state either never updates or resets
   every step, making layer output position-independent.
2. **Value-first sharing collapse** — layer 0's `v_first` becomes
   input-independent under gate saturation; all downstream layers
   inherit the same constant.
3. **STE residual interaction** — at init the inner path runs at full
   strength and the `residual_proj`-scaled skip is near-zero; if the
   inner path collapses, the skip can't rescue.

Localising which of the three requires layer-by-layer ablation and is
postponed; the fix is training-regime-level and does not depend on
knowing exactly which sub-pathology dominates.

## 4. Why gt-signals didn't collapse but SOTAB did

Same model, same optimizer, same learning rate:

|  | gt-signals-dbpedia | SOTAB-Schemaorg-CTA |
|---|---|---|
| Train samples | 1,999 | 116,887 |
| Epochs | 20 | 3 |
| Total gradient steps | ~2,500 | ~22,000 |
| Best val macro F1 | 0.126 | 0.0007 |
| Collapse | no | yes |

10× more gradient steps at the same aggressive hyperparameters is the
differentiator. The reference RWKV-LM training recipes in `ref/rwkv-lm/`
use lr ≈ 1e-4 with 1000+ warmup steps and always clip gradients. We
did none of that.

## 5. Reusable methodology

The diagnostic procedure generalises beyond SOTAB:

```bash
uv run --no-sync python scripts/sotab_diagnostic.py --max-val-samples 1500
```

Output at `build/diagnostics/{task}/{run_id}/` includes `report.md`
(human-readable), `summary.json` (machine-readable), and raw arrays
(embeddings, predictions, confusion, MCL clusters).

**When to run it**:

- After any "plateau" that doesn't look like it's converging.
- Before concluding that a training recipe "works."
- Before publishing any F1 number — the diagnostic confirms the
  number reflects actual learning rather than a collapsed mode
  prediction that happens to hit the base rate.

## 6. What the audit is *for* once we have a real representation

The MCL-inflation-sweep approach is van Dongen's core contribution:
clusters emerge as attractor basins of stochastic flow at different
granularities, controlled by one knob (inflation). Our use of it goes
beyond post-hoc evaluation:

- **Geometry audit** — does the learned embedding space admit
  recoverable hierarchical structure? If so, the classifier is a
  readout layer; if not, the representation needs more work.
- **DED-inference alternative to k-means** — for cross-table Data
  Element Discovery, MCL replaces k-means because we don't know the
  true number of data elements a priori. MCL finds them without being
  told.
- **Ontology-level evaluation** — multi-granularity purity reporting
  (leaf vs parent vs grandparent) is a more honest evaluator than
  flat leaf-level F1, matching the structure of the label space.

None of this is possible on a collapsed checkpoint. The audit is
postponed to after Stage B (pretraining) confirms we have a live
representation.
