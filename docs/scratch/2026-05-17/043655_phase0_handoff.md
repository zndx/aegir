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

Pending Block B's Phase 0 producing a usable backbone. Synthetic-SQL
+ counterfactual generators don't exist yet — write them Sunday
afternoon while Phase 0 finishes.

- `src/aegir/data/synth_table_qa.py` — grammar producing
  *"X is greater than the sum of Y when Z is K"* statements; binary
  truth label evaluated against the table.
- `src/aegir/data/synth_counterfactual.py` — take the surrounding
  `questions` text (TITLE/DESCRIPTION/SEGMENT_TEXT), swap one entity
  for another from the same column; binary "corrupted?" label.

Both generators consume the same parquet output as Phase 0.

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
