# Session: M2+ progress with A.1 validation on real data

Auto-mode session bridging M1 completion to M2 scope. User was wrapping up
other work; used the gap to land three committable workstreams and start a
20-epoch real-data validation run with the new SSD kernel.

## What shipped

### A.1 — Parallel EMA scan in DeChunkLayer via mamba-ssm SSD (`4a4953f`, `2b262f4`)

The sequential `_ema_scan` in `src/aegir/modules/dc.py` has been the dechunk hot
path since the H-Net port. Swap for `mamba_chunk_scan_combined` with the EMA →
SSM mapping `A=-1, dt=-log(d), B=(1-exp(-dt))/dt, C=1, initial_states=x[0]`.

Two surprises along the way:
1. **SSD kernel OOMs in backward at D≥256.** The Triton kernel is tuned for
   Mamba-2 shapes (many small heads); a single head with `headdim=D=384`
   requests ~328 KB shared mem vs RTX 4090's ~100 KB limit. Only surfaces in
   backward, only at realistic training shapes — my isolated micro-bench
   didn't trigger it. Fix: slice `D` into `nheads` of `headdim=64`
   (`AEGIR_DECHUNK_SSD_HEADDIM`).
2. **SSD regresses 2× at low `mean_F`.** Untrained routing gives `mean_F ≈ 0.001`
   so post-chunk `L` is ~2-10 tokens — way below SSD's break-even. Fix:
   dispatch sequential below `AEGIR_DECHUNK_SSD_MIN_L` (default 256).

Validated in isolation: fwd <1%, bwd <1% max-rel-err vs sequential in bf16.
Validated end-to-end: epoch 1 of the 20-epoch gt-signals-dbpedia run
completed in 283.8s with expected boundary diagnostics (stage0 mean_F=0.147,
stage1 mean_F=0.341) and rising F1 (0 → 0.006 macro at epoch 1).

Env knobs documented in `CLAUDE.md`. Tier-0 @gpu BDD lives in
`features/chunking/ema_scan.feature`.

### A.2 — ROSA step() fails loud instead of returning zeros (`4a4953f` commit)

`RWKV_ROSA.step()` silently returned `torch.zeros(...)` during RNN-mode
decoding — the worst failure mode for correctness. No current arch_layout
uses r/R blocks, so the bug was latent, but it would have quietly corrupted
any future inference path that wired ROSA.

Replaced with `NotImplementedError` + explanation pointing to w/W (RWKV-7
TimeMix) for step-mode decoding. A rolling-window fallback (replay last K
tokens through forward + extract tail) is documented as a future option if
we ever need ROSA for online generation.

Contract test: `features/inference/rosa_step.feature` asserts the
NotImplementedError is raised and mentions the w/W redirection.

### B.4 — AegirForDED head + SupCon loss + B-cubed F1 (`d8477e6`)

Cross-table Data Element Discovery is the M2 differentiator. This commit
lands the model + loss + metric contract so follow-on PRs can focus on the
dataset and training loop without re-litigating the shape API.

Head (`src/aegir/models/heads.py::AegirForDED`):
- Packed multi-column input via 2D `cls_indexes` (B, N_cols) with `-1` for
  padding rows.
- 2-layer MLP projector (standard contrastive pretraining pattern).
- Emits L2-normalized embeddings + `col_mask` so the loss can skip padding.

Loss (`src/aegir/utils/train.py::supcon_loss`):
- Supervised contrastive (Khosla et al., NeurIPS 2020) over the flat
  `(B*N, D)` matrix in one similarity pass.
- Same-cluster pairs are positives regardless of source sample — exactly the
  DED objective (columns sharing a data element cluster together).
- Returns 0 cleanly when no positive pairs exist.

Metric (`src/aegir/utils/train.py::bcubed_f1`):
- B-cubed precision/recall/F1 (Bagga & Baldwin, 1998). Standard for
  entity-resolution-style clustering evaluation.

End-to-end smoke (`scripts/train_ded_smoke.py`, `just ded-smoke`):
- Synthetic cross-table dataset with known cluster prefixes. Tiny model,
  3 epochs. Observed: B-cubed F1 rose 0.18 → 0.45 and train loss 3.9 → 1.7.
  The training loop works; the synthetic task is easy by design.

Tier-0 BDD: `features/ded/ded_head.feature` — four invariants covering
unit-norm output, padding zeroing, loss finiteness, and B-cubed F1 on a
perfect prediction.

## What's running

`CUDA_VISIBLE_DEVICES=1 AEGIR_DECHUNK_SCAN=ssd` gt-signals-dbpedia on small
model (~56M params), 20 epochs, max_length=1024, MMR=8, lr=3e-4. ~4.7 min
per epoch → ~95 min total. Log at `/tmp/bench_gt_small_ssd_v2.log`. Run
artifact at `outputs/runs/20260419T093635Z_2b262f4778_small_gt-signals-dbpedia/`.

## What's NOT done

- **A.3 Residual-init experiment**: skipped. `residual_proj.weight` is
  already zero-init and `STE.forward` returns ones unconditionally, so the
  inner-path contribution is at full strength at init. No clear hypothesis
  for change without evidence of an init-related convergence issue.
- **Real GitTables DED dataset**: download finished (562k parquets). Loader
  sanity check parked as task #32 — full-vocab discovery is an unknown time
  cost and needs to be bounded with `--max-tables` or a disk-backed cache
  (C.7 candidate).
- **A/B vs sequential on same seed**: would prove the SSD speedup claim on
  real data. Deferred — sequential run competes for CPU with MMR
  embedding, best to run back-to-back not concurrent.

## Next natural steps

1. Let the 20-epoch benchmark finish; read metrics.json + plots, record
   first "Aegir trains on real data with SSD" leaderboard row.
2. A/B the same config with `AEGIR_DECHUNK_SCAN=sequential` — measure
   wall-clock delta, validate the 1.20× end-to-end speedup extrapolated
   from synthetic shapes.
3. Write `GitTablesDEDDataset` — cross-table sampling with DBpedia type as
   cluster label. Wire into `TASK_DATASET_CLASSES` + a Justfile recipe.
4. Disk-backed MMR embedding cache (C.7) — one-time cost at first run,
   amortizes for every subsequent dataset scan.

## Files touched this session

```
src/aegir/modules/dc.py                              (A.1)
src/aegir/modules/rwkv.py                            (A.2)
src/aegir/models/heads.py                            (B.4)
src/aegir/utils/train.py                             (B.4)
features/chunking/ema_scan.feature                   (A.1 BDD)
features/chunking/step_defs/ema_scan_steps.py
features/inference/rosa_step.feature                 (A.2 BDD)
features/inference/step_defs/rosa_step_steps.py
features/ded/ded_head.feature                        (B.4 BDD)
features/ded/step_defs/ded_head_steps.py
features/steps/__init__.py
scripts/bench_ema_scan.py                            (A.1 micro-bench)
scripts/train_ded_smoke.py                           (B.4 smoke trainer)
Justfile                                             (ded-smoke recipe)
CLAUDE.md                                            (env knobs + session-era updates)
docs/notes/2026-04-19/091337_a1_ema_scan_ssd.md      (A.1 engineering note)
docs/notes/2026-04-19/094204_session_m2_progress.md  (this file)
```

## Memory touched

- `dechunk_ssd_kernel.md` (new) — SSD kernel shape constraints, OOM fix,
  env knobs.
- `MEMORY.md` — index entry + pitfall entry pointing at the new note.

## BDD state

- `just bdd-0`: 10 features, 30 scenarios passing, 0 failed. ~60 s.
- `just bdd-1`: 10 features, 32 scenarios passing, 0 failed. ~90 s.
- No @slow run performed this session (would need 5+ min plus fixture
  training; tier-1 coverage is adequate for this PR set).
