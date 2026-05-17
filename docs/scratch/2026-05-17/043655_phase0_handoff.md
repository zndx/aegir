# Phase 0 pretraining — Sunday morning handoff

**Block A complete (Sat 21:30 → ~22:40 MDT).** Infrastructure for the
v0.2 dense-text + table-text pretraining pass is in-repo and ready to
launch. Block B (full Phase 0 pretrain) is the Sunday morning
priority.

## What's in-repo (committed tonight)

| Path | Purpose |
|---|---|
| `src/aegir/data/tokenizer.py` | Added `CELL_BOUNDARY_TOKEN_ID = 5` to the reserved-special range |
| `src/aegir/data/serialization.py` | `cell_boundary_sentinel: bool` kwarg on `serialize_table` — when True, emits the new sentinel byte before each cell value (TAPAS-style explicit prior) |
| `src/aegir/data/table_dataset.py` | Plumbs `cell_boundary_sentinel` through `_SotabDatasetBase.__init__` |
| `train.py` | Adds `--cell-boundary-sentinel` CLI flag |
| `scripts/protos/interaction.proto` | Vendored from TAPAS (Apache 2.0) |
| `scripts/protos/interaction_pb2.py` | `protoc`-generated bindings |
| `scripts/tapas_proto_to_aegir.py` | Reads `interactions.txtpb.gz` → emits parquet of (table_id, byte_ids, length). Uses BOS/SEP/BOUNDARY/EOS specials. Filters <2 col / <2 row tables. |
| `train_pretrain.py` | Pretraining entry point — `AegirForCausalLM` head, byte-level next-token CE, held-out perplexity eval, boundary diagnostics + metrics.json + checkpoint on val improvement |
| `scripts/split_pretrain_parquet.py` | Deterministic seeded train/val split for the pretrain parquet |
| `src/aegir/data/synth_table_qa.py` | TAPAS-style synthetic SQL grammar — 3 statement patterns (A: agg op constant, B: cell op constant, C: agg op agg), 5 aggregations (SUM/MIN/MAX/AVG/COUNT), 6 operators with multiple verbalizations, rejection-sampled label balance |
| `scripts/generate_synth_sql.py` | Phase 0.5 corpus driver — reads TAPAS interactions.txtpb.gz → emits (table_bytes, statement_bytes, label) parquet. Smoke: ~4 examples/table, 50/50 label balance |
| `src/aegir/data/synth_counterfactual.py` | TAPAS-style counterfactual generator — finds cell values that appear as substrings of surrounding text (word-boundary aware, min-len 4) and swaps one for a different value from the same column. Original→False, perturbed→True. |
| `scripts/generate_synth_counterfactual.py` | Phase 0.5 corpus driver for counterfactuals. Smoke: ~712 examples/5K records, 50/50 label balance, diverse cross-column swaps |
| `src/aegir/data/tokenizer.py` | Added `TRUE_TOKEN_ID=6`, `FALSE_TOKEN_ID=7` so binary labels can be expressed as single tokens — reuses LM head, no classification head needed |
| `src/aegir/data/phase05.py` | `Phase05Dataset` — reads synth-SQL + counterfactual parquets and emits `[BOS] table [SEP] second [SEP] [LABEL] [EOS]` byte sequences for `train_pretrain.py` |
| `train_pretrain.py` (extended) | Now accepts `--phase05-synth-sql` / `--phase05-counterfactual` flags as alternative to `--train-parquet`/`--val-parquet`. Same loss, same boundary diagnostics, same checkpoint logic. |

## What's on-disk (not in git)

| Path | Contents | Status |
|---|---|---|
| `/raid/datasets/tapas/interactions.txtpb.gz` | TAPAS 2020-05 dump, 2.0 GB, 6.35M records | Downloaded, md5 verified |
| `/raid/datasets/tapas/aegir_bytes_v0.parquet` | Full byte-level pretrain corpus | Generating overnight (~90 min, ~14% keep rate → ~890K rows expected) |
| `/raid/checkpoints/aegir-cellboundary-ab/{baseline,sentinel}` | A1 A/B run outputs | Baseline launched; sentinel needs to be kicked off after baseline completes |

## Where the cell-boundary A/B stands

| Run | Status | Output dir |
|---|---|---|
| Baseline (no sentinel) | running, tiny model, 1 epoch SOTAB-CTA | `/raid/checkpoints/aegir-cellboundary-ab/baseline/runs/<id>/metrics.json` |
| Sentinel (with `--cell-boundary-sentinel`) | NOT YET LAUNCHED | needs to be kicked off |

When baseline epoch 1 lands, kick off sentinel:
```bash
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync torchrun \
  --nproc_per_node=6 --master_port=29502 train.py \
    --task sotab --data-dir /raid/datasets/sotab \
    --model-size tiny --epochs 1 --batch-size 16 --lr 1e-4 \
    --warmup-ratio 0.05 --weight-decay 0.01 \
    --max-length 128 --max-context-cols 8 \
    --num-workers 4 --log-interval 100 \
    --cell-boundary-sentinel \
    --output-dir /raid/checkpoints/aegir-cellboundary-ab/sentinel \
    2>&1 | tee /tmp/aegir-ab-sentinel.log
```

Compare `metrics.json` boundary stats from both runs. Signal we're looking for: does the sentinel sharpen the chunker's `mean_F` distribution (more bimodal) or shift `achieved_N` closer to target 2.0?

## Block B — Phase 0 launch (Sunday morning)

Assuming `aegir_bytes_v0.parquet` finished overnight:

```bash
# 1. Split parquet into train/val (95/5) — write a small helper if needed:
#    /raid/datasets/tapas/aegir_bytes_v0_train.parquet
#    /raid/datasets/tapas/aegir_bytes_v0_val.parquet

# 2. Launch Phase 0:
cd /home/rch/local/src/zndx/aegir
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync torchrun \
  --nproc_per_node=6 --master_port=29503 train_pretrain.py \
    --train-parquet /raid/datasets/tapas/aegir_bytes_v0_train.parquet \
    --val-parquet /raid/datasets/tapas/aegir_bytes_v0_val.parquet \
    --model-size small --epochs 8 --batch-size 8 --lr 1e-4 \
    --warmup-ratio 0.05 --weight-decay 0.01 \
    --max-length 1024 --num-workers 4 --log-interval 50 \
    --output-dir /raid/checkpoints/aegir-pretrain-v0.2 \
    2>&1 | tee /tmp/aegir-pretrain-v0.2.log
```

### Throughput envelope (re-derive on Sunday with actuals)

Estimated from SOTAB run: ~123K tokens/sec on 6 GPUs at bf16. With
max_length=1024 and batch_size=8/GPU, that's:
- tokens/step = 1024 × 8 × 6 = ~49K
- step/sec ≈ 2.5
- step/sec × tokens/step = ~123K tokens/sec (matches)
- ~440M tokens/hour

If kept dataset is 890K rows × ~860 tokens/row ≈ 770M tokens, then
one full epoch ≈ 1h45m. 8 epochs ≈ 14 hours. Comfortably fits in
the Sunday window if started by ~10 MDT.

### What to watch

1. **Chunker boundary stats per epoch.** Run A's lesson: stage-1 mean_F
   collapses to ~0.05 if LR is too high. We're at lr=1e-4 which kept it
   stable on SOTAB; should transfer, but verify on epoch 1. If
   `stage1_abs_ratio_err > 0.20` at epoch 1, lower LR to 5e-5.
2. **Val perplexity trajectory.** Expect rapid drop epoch 1
   (4.5 → 3.5 nats ballpark for byte-level on structured text),
   then slower. Random baseline = `ln(65536) = 11.09` nats. Real
   English text byte-level ppl converges around 1.3-1.5 nats with
   small models, so anything < 3.0 by epoch 8 is healthy.
3. **NaN/Inf in logits.** Bf16 + tied embeddings + 65K vocab is
   numerically tight. If loss spikes, check grad-norm; we clip at 1.0
   but a sudden grad explosion would still propagate one step.

## Block C — Phase 0.5 (intermediate pretrain, TAPAS-style)

**Synth-SQL grammar + corpus driver are committed.** Generation can
run during/after Phase 0 on a held-back CPU while GPUs train Phase 0:

```bash
LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \
    scripts/generate_synth_sql.py \
    --input /raid/datasets/tapas/interactions.txtpb.gz \
    --output /raid/datasets/tapas/aegir_synth_sql_v0.parquet \
    --per-table 4 --seed 4649
```

Expected: ~3.5M examples (smoke shows ~4 examples / usable table, ~700
usable tables per 5K input records). ~50 min wall on a single CPU core.

The trainer for Phase 0.5 isn't built yet — needs to:
1. Concatenate (table_bytes + statement_bytes) into a single input.
2. Decide modality: (a) append `True`/`False` and train as LM continuation;
   (b) classification head with 2 labels. (a) reuses train_pretrain.py
   with minimal change; (b) reuses train.py with `--num-classes 2`.

**Both Phase 0.5 generators + trainer wiring landed tonight.** Sunday
afternoon execution is just `torchrun train_pretrain.py --phase05-*`
once the corpora are generated:

```bash
# Step 1 — generate both Phase 0.5 corpora (CPU, ~50 min each in parallel)
LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \
    scripts/generate_synth_sql.py \
    --input /raid/datasets/tapas/interactions.txtpb.gz \
    --output /raid/datasets/tapas/aegir_synth_sql_v0.parquet \
    --per-table 4 --seed 4649 &

LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \
    scripts/generate_synth_counterfactual.py \
    --input /raid/datasets/tapas/interactions.txtpb.gz \
    --output /raid/datasets/tapas/aegir_synth_cf_v0.parquet \
    --pairs-per-table 2 --seed 4649 &

wait

# Step 2 — Phase 0.5 training (after Phase 0 produces a backbone)
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync torchrun \
  --nproc_per_node=6 --master_port=29504 train_pretrain.py \
    --phase05-synth-sql /raid/datasets/tapas/aegir_synth_sql_v0.parquet \
    --phase05-counterfactual /raid/datasets/tapas/aegir_synth_cf_v0.parquet \
    --model-size small --epochs 4 --batch-size 8 --lr 5e-5 \
    --warmup-ratio 0.05 --weight-decay 0.01 \
    --max-length 1024 --num-workers 4 --log-interval 50 \
    --output-dir /raid/checkpoints/aegir-phase05-v0.2 \
    --resume-from /raid/checkpoints/aegir-pretrain-v0.2/runs/<phase0-run-id>/best_model.pt \
    2>&1 | tee /tmp/aegir-phase05-v0.2.log
```

(The `--resume-from` flag doesn't exist yet — would need a small
addition to load state_dict from a Phase 0 checkpoint before training
starts. Easy 10-line patch Sunday morning, or skip it and re-init,
losing Phase 0's learning. Recommend implementing it before launch.)

Eval semantics for Phase 0.5: condition on the prefix up to the second
`[SEP]`, examine `argmax(logits[TRUE_TOKEN_ID], logits[FALSE_TOKEN_ID])`,
compare to label. Could add this as a dedicated eval mode in
`train_pretrain.py:evaluate` but the byte-level ppl on the full sequence
is also a reasonable proxy in the short term.

## Scale strategy — small → 2x "fire for effect"

After small (~56M) Phase 0 validates the pipeline, the recommended 2x
config for the production checkpoint:

```
d_model       = [384, 512, 512]      # 1.5x hidden vs small
arch_layout   = ["w6", ["w6", ["w12"], "w6"], "w6"]   # 1.5x depth
params        = ~120M  (~2.1x small)
```

Or depth-only variant (~110M):
```
d_model       = [256, 384, 384]      # same as small
arch_layout   = ["w8", ["w8", ["w16"], "w8"], "w8"]   # 2x depth
```

Hold fixed across both: `lr=1e-4` (chunker-safe), `max_length=1024`,
`λ_lb=0.1`, `downsample_factor=2.0`, bf16 AMP, cosine + warmup 5%.

Green-light to 2x once small Phase 0 shows: (a) per-epoch
`stage1_abs_ratio_err < 0.10` (chunker stable), (b) val ppl trending
below the random baseline `e^11.09 ≈ 65,536`, (c) no NaN spikes.

Estimated wall: small 8-epoch ≈ 14h; 2x 8-epoch ≈ 28h (or 4 epochs in
14h for same token budget). Both fit before Monday close if small
kicks off Sunday morning.

## Decision points before sleeping tonight

- [ ] If A1 baseline completes before 23:30 MDT, kick off the sentinel
      run. If not, leave both for tomorrow.
- [ ] If the TAPAS conversion finishes (~06:00 UTC ≈ 00:00 MDT), good;
      otherwise it'll keep running and we'll catch up at wake.
- [ ] No need to launch Phase 0 tonight — better to start fresh on
      Sunday morning when we can watch the first epoch closely.

## Risks parked

- `train_pretrain.py` smoke test is still incomplete — CPU smoke
  confirmed imports + model build + parquet load + collate shapes,
  but the forward pass needs CUDA (fla `chunk_rwkv7` is Triton/GPU
  only). Run this single-GPU smoke first thing Sunday before
  launching the multi-GPU pretrain — should take ~30s wall-clock:
  ```bash
  cd /home/rch/local/src/zndx/aegir
  LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python train_pretrain.py \
    --train-parquet /tmp/aegir_bytes_smoke.parquet \
    --val-parquet /tmp/aegir_bytes_smoke.parquet \
    --model-size tiny --epochs 1 --batch-size 4 --lr 1e-4 \
    --max-length 512 --num-workers 0 --log-interval 5 \
    --output-dir /tmp/aegir-pretrain-smoke
  ```
  Pass criteria: loss decreases over the first few steps, val ppl
  drops below the random baseline `exp(ln(65536)) ≈ 65536`, no NaN
  in train_loss / val_loss, boundary diagnostics emit non-zero
  values for stage0 and stage1.
- Pyright noise on torch private imports — pre-existing, not introduced
  tonight, not a real bug.
- TAPAS proto bindings (`scripts/protos/interaction_pb2.py`) are
  protoc-generated; they're checked in but rebuildable by running
  `protoc --python_out=. interaction.proto` from `scripts/protos/`.
