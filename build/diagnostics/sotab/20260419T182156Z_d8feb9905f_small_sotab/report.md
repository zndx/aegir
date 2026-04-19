# SOTAB diagnostic — 20260419T182156Z_d8feb9905f_small_sotab

## 1. Prediction distribution

- N val samples: 1500
- Distinct predicted classes: 1 / 91
- Exact-match accuracy: 0.0327
- Top-1 fraction: 1.000
- **Mode collapse?** yes (threshold: top-1 class ≥ 50% of preds)

| Rank | Label | Count | Fraction |
|------|-------|-------|----------|
| 1 | `currency` | 1500 | 1.000 |

## 2. Cluster geometry

- Samples used: 1500
- Embedding energy (mean ||x||): 6.9839
- Per-dim variance: max 2.96e-06, median 4.51e-08
- Max pairwise L2 (50-sample probe): 0.020387 (ratio to energy: 2.92e-03)

> **COLLAPSE DETECTED**: max pairwise L2 across 50 random val
> samples is below 1e-3 — the pre-classifier embedding is
> effectively **constant across inputs**. This is representation
> collapse, not weak clustering. Subsequent intra/inter and MCL
> numbers are mathematically degenerate (every pair is 'close'
> because every pair IS identical).

- Intra-class mean cosine distance: 0.0000 (median 0.0000, n_pairs=20428)
- Inter-class mean cosine distance: 0.0000 (median 0.0000, n_pairs=1103822)
- **intra/inter ratio not meaningful** under representation collapse.

## 3. MCL inflation sweep

- Samples used: 1500

| Inflation | # Clusters | Purity (leaf) | Purity (parent) |
|-----------|------------|---------------|-----------------|
| 1.4 | 1 | 0.0333 | 0.8513 |
| 2.0 | 1 | 0.0333 | 0.8513 |
| 3.0 | 1 | 0.0333 | 0.8513 |
| 4.0 | 1 | 0.0333 | 0.8513 |

### Interpretation

- MCL produces 1 cluster at every inflation because the embedding
  space has collapsed to a single point. The reported
  `purity_parent` ≈ 0.851 is just the
  fraction of val samples whose label rolls up to the *modal*
  Schema.org parent — a property of the label distribution, NOT
  of the embedding geometry.
- **Verdict: geometry-audit cannot proceed on a collapsed model.**
  Fix the collapse first (gradient-flow hygiene), then re-run.
