# train_pretrain.py GPU smoke — 1 epoch on 632-sample smoke parquet

**Date:** 2026-05-17
**Command:**
```
LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python train_pretrain.py \
  --train-parquet /tmp/aegir_bytes_smoke_train.parquet \
  --val-parquet /tmp/aegir_bytes_smoke_val.parquet \
  --model-size tiny --epochs 1 --batch-size 4 --lr 1e-4 \
  --max-length 512 --num-workers 0 --log-interval 5 \
  --output-dir /tmp/aegir-pretrain-smoke
```

## Pipeline pass criteria (all met ✅)

- Forward pass works on GPU (no NaN, no Triton failure, no fla shape mismatch).
- Loss decreases meaningfully from random init: step 80 ≈ 10.27, step 90 ≈ 9.24,
  step 120 ≈ 8.70 (lowest). Random baseline = `ln(65536) ≈ 11.09`.
- `best_model.pt` saved on val improvement.
- Throughput ~4.2 step/s on single GPU (tiny model, batch_size=4,
  max_length=512).
- `metrics.json` populated with full schema including boundary stats.

## Final metrics

| metric | value |
|---|---:|
| train_loss (epoch mean) | 13.78 (warmup-skewed) |
| val_loss | 9.37 |
| val_ppl | 11,743 |
| val_tokens | 34,229 |
| wall_seconds | 94.5 |

Val ppl=11,743 vs random-uniform baseline ppl=65,536 → ~6× better than
uniform; ~1.7 nats of token-level entropy captured, which is consistent
with "learned the marginal byte distribution of UTF-8 English text and
not much else" — exactly what 158 update steps on 632 samples should
produce.

## ⚠️ Chunker calibration on the smoke is bad

| stage | mean_F | achieved_N | abs_ratio_err |
|---|---:|---:|---:|
| stage0 | 0.349 | 8.4 | 0.159 |
| stage1 | **0.014** | **197.1** | **0.486** |

Stage-1 mean_F=0.014 means essentially no boundaries are being flagged
in the deepest stage — the routing module is producing near-zero
boundary probability. The load-balancing loss is high (1.14 vs
target ~1.0) but hasn't pushed the routing back toward 0.5 in 158
steps.

**This is almost certainly an artifact of the smoke's scale** rather
than a fundamental issue:

- Total update steps = 158 (632 samples / batch 4). Warmup + cosine
  decay leaves only ~150 effective steps, with the LR averaging
  ~5e-5 across them.
- The STE-binarized boundary indicator has limited gradient signal
  early on — it needs many steps to drift into equilibrium.
- The cell-boundary A/B (which ran on SOTAB with ~1200 steps per
  epoch and a *cleaner* serialization) showed stage1 achieved_N
  converging to 2.0 in a single epoch. So the chunker *can*
  stabilize at this scale — but it needs more steps than 158, or
  more samples, or both.

## What this means for Sunday's Phase 0 launch

The smoke validates the *pipeline*. It does NOT validate that the
chunker will stabilize on the full corpus. Critical first check on
Sunday morning is the epoch-1 metrics.json:

- **Good signal (continue):** stage1 abs_ratio_err < 0.10 by end of
  epoch 1. Phase 0 trains as planned.
- **Bad signal (intervene):** stage1 abs_ratio_err > 0.20 or
  achieved_N > 5 at epoch 1. The architecture isn't stable at scale
  even with lr=1e-4 — try (a) raise `λ_lb` to 0.3 to push the
  chunker harder, (b) warm-start with `λ_lb=0.5` and decay to 0.1,
  or (c) re-examine the routing module gradient flow.

Most likely outcome: chunker stabilizes once it sees ~5K+ update
steps on real data with realistic table structure (which the
smoke parquet has — 702 records of real Wikipedia table-text). Worth
running 200-300 *more* steps of smoke on the same data to verify
this hypothesis before tomorrow's launch.

## Artifacts

- `/tmp/aegir-pretrain-smoke/runs/20260517T...Z_*_tiny_pretrain/`
- Training log: `/tmp/aegir-pretrain-smoke.log`
