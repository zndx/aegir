# A.1: EMA scan → SSD kernel (DeChunkLayer)

## Problem

`src/aegir/modules/dc.py::_ema_scan` was a pure-Python loop that appended per-step
outputs to a list and `torch.stack`-ed them. At `L=2048`, D=192, B=32, this was
~80ms/fwd and ~300ms/bwd (autograd unrolls the list append into L sequential ops).
Each DeChunkLayer call in the recursive hierarchy triggers one scan; with
`arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]` there are 2 dechunk calls per
forward pass, amplifying the cost.

## Approach

Swap the sequential scan for mamba-ssm's `mamba_chunk_scan_combined` (SSD kernel,
Triton) with the following mapping of the EMA recurrence
`y[t] = d[t] * y[t-1] + (1 - d[t]) * x[t]` to the SSM form
`h[t] = exp(dt*A) * h[t-1] + dt * B * x[t]`:

| EMA                         | SSD                                           |
|-----------------------------|-----------------------------------------------|
| `A = -1` (scalar)           | `A` shape `(nheads=1,)`                       |
| `dt[t] = -log(d[t])`        | `dt[0] = 0` + `initial_states = x[0]` for `y[0] = x[0]` |
| `B[t] = (1 - exp(-dt)) / dt`| time-varying, shape `(B, L, 1, 1)`            |
| `C = 1`                     | ones, shape `(B, L, 1, 1)`                    |
| `x`                         | `x.unsqueeze(2)`, shape `(B, L, nheads=1, headdim=D)` |
| state dim                   | `ngroups=1, dstate=1`                         |

The `dt[0] = 0` trick (with `initial_states = x[0]`) preserves the sequential
reference's convention that `y[0] = x[0]` (untouched by EMA). In the real
DeChunkLayer workflow `boundary_prob[:, 0]` is always forced to 1.0 by the
RoutingModule's leading pad, so `decay[:, 0] ≈ 1e-4` after the clamp — the
`initial_states` override is cleaner than relying on numerical near-equivalence.

## Numerical equivalence

Validated in two ways:

1. `scripts/bench_ema_scan.py` micro-bench: SSD vs sequential @ bf16,
   `B=32, L=512, D=192`:
   - forward max-rel-err: **0.92%**
   - backward (input gradient) max-rel-err: **1.07%**
2. `features/chunking/ema_scan.feature` (tier-0 @gpu): asserts forward <2%,
   backward <3% with RoutingModule-style first-token pad.

Both well inside bf16 precision (which is already the FLA kernel precision floor
for RWKV-7 TimeMix).

## Performance

Micro-bench (isolated, `B=32, L=512, D=192`, bf16 on RTX 4090):

|               | fwd (ms) | fwd+bwd (ms) |
|---------------|----------|--------------|
| sequential    | 39.5     | 108.3        |
| torch.compile | 3.1      | — (autograd still O(L))
| SSD           | 2.3      | 7.1          |
| logspace ✗    | 0.5      | 1.4          (NaN — numerically unstable)
| assoc_scan ✗  | 0.5      | —            (no autograd support in torch 2.6) |

**End-to-end training smoke, synthetic data**, `--model-size tiny`, `L=2048`,
`B=32`, 20 train steps:

|                              | wall time | speedup |
|------------------------------|-----------|---------|
| sequential                   | 94.2 s    | 1.00×   |
| SSD (unconditional)          | 203.8 s   | 0.46× (regression — setup overhead on short post-chunk L) |
| SSD with `L ≥ 256` threshold | 78.5 s    | **1.20×** |

The untrained boundary predictor produces extremely sparse boundaries
(`mean_F ≈ 0.001`), so the post-chunk sequence length the EMA sees at inner
stages is just ~2-10 tokens. Below `L=256` SSD's `chunk_size=64` padding +
reshape + kernel launch cost dominates, hence the threshold.

## Threshold selection

`AEGIR_DECHUNK_SSD_MIN_L` env var (default 256). Rationale: measured crossover
point on RTX 4090. Tunable per-device; 256 is conservative — on trained models
with denser boundaries, the real post-chunk L is higher, so the threshold fires
more often and the speedup grows.

## Public surface

- `AEGIR_DECHUNK_SCAN` = `"auto"` (default) | `"sequential"` | `"ssd"`
- `AEGIR_DECHUNK_SSD_MIN_L` = integer (default 256)

Default behavior: auto → SSD when CUDA + mamba-ssm importable, threshold-gated.
CPU-only machines and tier-0 BDD runs transparently use sequential.

## What still isn't fast

When `mean_F → 0` (untrained model) the scan is effectively free regardless of
backend — the real bottleneck is elsewhere. As training progresses and
`mean_F` rises toward the load-balancing target (~0.33 for downsample_factor=3),
the scan will see longer post-chunk L and SSD will win more.

The larger structural cost at long `L_original` is the **pre-chunking** isotropic
encoder which still runs at full L=2048 — that's an RWKV-7 TimeMix, already
kernel-accelerated via fla. Not a target for this ticket.

## Tier-1 BDD

`just bdd-1` green: 8 features, 27 scenarios, 0 failed, 1:29s. Includes new
`features/chunking/ema_scan.feature` scenarios plus regression kill-test of
the existing training-loop scenarios.

## Not done / punted

- Fair A/B on a **trained** model with dense boundaries (mean_F > 0.1) — needs
  a longer run (100+ steps) on a real task. Parked until GitTables finishes
  downloading; then `benchmarks-reveal-match` provides natural ground.
- torch.compile wrapping of sequential — irrelevant once SSD path is active
  for long L; sequential handles tiny-L better than compiled would anyway.
- SSD with higher `dstate` (> 1) — no gain for scalar EMA; would only matter
  if we ever needed per-feature decay.
