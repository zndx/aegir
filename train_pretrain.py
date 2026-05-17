"""Aegir byte-level pretraining on TAPAS-style table-text corpora.

Variant of ``train.py`` for the LM pretraining objective:
- ``AegirForCausalLM`` head with tied embeddings (no classification head).
- Loss: byte-level next-token cross-entropy + λ_lb load-balancing on the
  hierarchical chunker.
- Eval: held-out perplexity.
- Dataset: parquet of (table_id, byte_ids, length) rows produced by
  ``scripts/tapas_proto_to_aegir.py``.

Usage (multi-GPU, full)::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync torchrun \\
        --nproc_per_node=6 train_pretrain.py \\
        --train-parquet /raid/datasets/tapas/aegir_bytes_v0.parquet \\
        --val-parquet /raid/datasets/tapas/aegir_bytes_v0_val.parquet \\
        --model-size small --epochs 8 --batch-size 8 --lr 1e-4 \\
        --max-length 1024 --output-dir /raid/checkpoints/aegir-pretrain-v0.2

Single-GPU smoke (synthetic-data-free)::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python train_pretrain.py \\
        --train-parquet /tmp/aegir_bytes_smoke.parquet \\
        --val-parquet /tmp/aegir_bytes_smoke.parquet \\
        --model-size tiny --epochs 1 --batch-size 4 --lr 1e-4 \\
        --max-length 512 --output-dir /tmp/aegir-pretrain-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from aegir.data.tokenizer import TRUE_TOKEN_ID, FALSE_TOKEN_ID
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForCausalLM
from aegir.utils.train import (
    boundary_diagnostics,
    group_params,
    load_balancing_loss,
)


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                 label_token_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position cross-entropy with label-token up-weighting.

    Phase 0.5 sequences have ~1020 byte-position targets and 1
    TRUE/FALSE label-token target. Mean CE would put <0.1% of the
    gradient on the actual classification objective. We up-weight
    positions whose target is TRUE_TOKEN_ID or FALSE_TOKEN_ID by
    ``label_token_weight``; padded positions (ignore_index=-100)
    contribute nothing.

    Returns (weighted_mean_loss, label_only_loss, n_label_positions).
    label_only_loss is the unweighted mean over the label positions
    alone — the headline Phase 0.5 metric. n_label_positions is the
    count of label positions in the batch (useful for averaging
    across batches).
    """
    B, L, V = logits.shape
    flat_logits = logits.reshape(B * L, V)
    flat_labels = labels.reshape(B * L)
    per_pos = nn.functional.cross_entropy(
        flat_logits, flat_labels,
        ignore_index=-100, reduction="none",
    )
    valid = (flat_labels != -100).float()
    is_label = ((flat_labels == TRUE_TOKEN_ID) |
                (flat_labels == FALSE_TOKEN_ID)).float()

    if label_token_weight == 1.0:
        # Vanilla path — equivalent to mean CE with ignore_index.
        weighted = (per_pos * valid).sum() / valid.sum().clamp(min=1.0)
    else:
        weights = valid * (1.0 + (label_token_weight - 1.0) * is_label)
        weighted = (per_pos * weights).sum() / weights.sum().clamp(min=1.0)

    label_loss_sum = (per_pos * is_label).sum()
    n_label = is_label.sum()
    label_only = label_loss_sum / n_label.clamp(min=1.0)
    return weighted, label_only, n_label


# ─────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────

class ParquetByteDataset(Dataset):
    """Rank-sharded lazy parquet reader for (byte_ids, length) rows.

    Each DDP rank loads ONLY its 1/world_size shard from the parquet
    via row-groups. With 2.1M rows / 6 ranks = ~350K rows/rank,
    in-memory footprint per rank drops from ~14 GB (full table) to
    ~2.3 GB.

    Sample access is lazy: each ``__getitem__`` materializes one row's
    list[int]; nothing is held except the row's transient Python
    list.

    History: a previous attempt that did ``to_pylist()`` on the full
    table per rank exhausted RAM (28 bytes/Python-int × 1.8 B tokens
    = ~50 GB/rank) and triggered a system OOM cascade after 6+ hours.
    Then a version that did full-table load per rank (no to_pylist)
    still came to ~14 GB/rank × 6 = 84 GB which collides with DataLoader
    fork overhead. Row-group sharding fixes both.

    Use ``RandomSampler`` (not ``DistributedSampler``) with this
    dataset since each rank already holds a disjoint shard.
    """

    def __init__(self, path: str | Path, max_length: int,
                 rank: int = 0, world_size: int = 1):
        pf = pq.ParquetFile(str(path), memory_map=True)
        n_rg = pf.num_row_groups
        # Round-robin row-group assignment so each rank gets roughly
        # equal share even when n_rg isn't divisible by world_size.
        my_rgs = list(range(rank, n_rg, world_size))
        self.table = pf.read_row_groups(my_rgs, columns=["byte_ids", "length"])
        self._byte_ids = self.table.column("byte_ids")  # ChunkedArray, lazy
        self.lengths = self.table.column("length").to_numpy()
        self.max_length = max_length
        self._n = self.table.num_rows
        self.rank = rank
        self.world_size = world_size
        self.n_row_groups_local = len(my_rgs)
        self.n_row_groups_global = n_rg

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        ids = self._byte_ids[idx].as_py()[: self.max_length]
        return torch.tensor(ids, dtype=torch.long)


def lm_collate_fn(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch; build input/label/mask.

    For LM training: input = seq[:-1], labels = seq[1:]. Padding positions
    get label_id = -100 so they're ignored by CrossEntropyLoss(ignore_index=-100).
    """
    max_len = max(t.shape[0] for t in batch)
    B = len(batch)
    PAD = 0  # PAD_TOKEN_ID — aligned with ByteTokenizer specials.

    full = torch.full((B, max_len), PAD, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, t in enumerate(batch):
        L = t.shape[0]
        full[i, :L] = t
        mask[i, :L] = True

    # Shift for next-token prediction. input length = max_len - 1.
    input_ids = full[:, :-1].contiguous()
    target_ids = full[:, 1:].contiguous()
    valid = mask[:, 1:].contiguous()  # don't supervise padding positions
    labels = target_ids.masked_fill(~valid, -100)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask": mask[:, :-1].contiguous(),
    }


# ─────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────

def make_pretrain_model(args) -> AegirForCausalLM:
    """Build an AegirForCausalLM at the requested model size."""
    if args.model_size == "tiny":
        config = AegirConfig(
            arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
            d_model=[128, 192, 192],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[2, 3, 3],
                                rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=0,
            task_type="lm",
            tie_embeddings=True,
        )
    elif args.model_size == "small":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
            d_model=[256, 384, 384],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[4, 6, 6],
                                rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=0,
            task_type="lm",
            tie_embeddings=True,
        )
    elif args.model_size == "base":
        config = AegirConfig(
            arch_layout=["w4", ["w4", ["w12"], "w4"], "w4"],
            d_model=[768, 1024, 1024],
            d_intermediate=[0, 0, 0],
            vocab_size=args.vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[12, 16, 16],
                                rotary_emb_dim=[48, 64, 64], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=0,
            task_type="lm",
            tie_embeddings=True,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    model = AegirForCausalLM(config)
    model.init_weights()
    return model


# ─────────────────────────────────────────────────────────────────────────
# Training / eval loops
# ─────────────────────────────────────────────────────────────────────────

def _accumulate(accum: dict[str, float], diag: dict[str, float]) -> None:
    for k, v in diag.items():
        accum[k] = accum.get(k, 0.0) + v


def _finalize(accum: dict[str, float], n: int) -> dict[str, float]:
    if n == 0:
        return {}
    return {k: v / n for k, v in accum.items()}


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, args,
                epoch_idx: int = 0, is_main: bool = True):
    """One pretrain epoch — next-token CE (optionally label-weighted) + load-balancing."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_label_loss = 0.0
    total_lb_loss = 0.0
    total_label_positions = 0
    n_label_batches = 0
    n_batches = 0
    n_tokens = 0
    boundary_accum: dict[str, float] = {}

    log_interval = getattr(args, "log_interval", 20) or 20
    label_w = getattr(args, "label_token_weight", 1.0) or 1.0
    epoch_start = time.time()
    window_start = epoch_start
    window_loss = 0.0
    window_count = 0

    for step_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            output = model(input_ids, mask=mask)
            logits = output.logits

            task_loss, label_only_loss, n_label = _weighted_ce(
                logits, labels, label_token_weight=label_w,
            )

            lb_loss = torch.tensor(0.0, device=device)
            if output.bpred_output:
                for bpred in output.bpred_output:
                    lb_loss = lb_loss + load_balancing_loss(bpred, N=args.downsample_factor)
                lb_loss = lb_loss / len(output.bpred_output)

            loss = task_loss + args.lambda_lb * lb_loss

        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_lb_loss += lb_loss.item()
        n_label_f = float(n_label.item())
        if n_label_f > 0:
            total_label_loss += label_only_loss.item()
            total_label_positions += int(n_label_f)
            n_label_batches += 1
        n_batches += 1
        n_tokens += int((labels != -100).sum().item())

        _accumulate(
            boundary_accum,
            boundary_diagnostics(output.bpred_output, target_N=args.downsample_factor),
        )

        window_loss += loss.item()
        window_count += 1
        if is_main and log_interval > 0 and (step_idx + 1) % log_interval == 0:
            now = time.time()
            window_elapsed = now - window_start
            steps_per_sec = window_count / max(window_elapsed, 1e-9)
            cur_lr = optimizer.param_groups[0]["lr"]
            avg_loss = window_loss / window_count
            label_loss_str = (
                f" label_loss={label_only_loss.item():.4f}"
                if n_label_f > 0 else ""
            )
            print(
                f"  [e{epoch_idx} step {step_idx+1:6d}]  "
                f"loss={avg_loss:.4f}  ppl={math.exp(min(avg_loss, 20)):.2f}{label_loss_str}  "
                f"lr={cur_lr:.2e}  "
                f"{steps_per_sec:.2f} step/s ({window_elapsed:.1f}s for {window_count} steps)",
                flush=True,
            )
            window_start = now
            window_loss = 0.0
            window_count = 0

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_task_loss": total_task_loss / max(n_batches, 1),
        "train_label_loss": (
            total_label_loss / n_label_batches
            if n_label_batches > 0 else None
        ),
        "train_lb_loss": total_lb_loss / max(n_batches, 1),
        "boundary": _finalize(boundary_accum, n_batches),
        "n_tokens": n_tokens,
        "n_label_positions": total_label_positions,
        "wall_seconds": time.time() - epoch_start,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, args):
    """Held-out byte-level perplexity + (when present) label-position metrics.

    Reports four metrics:
    - ``val_loss``: unweighted mean CE over all valid positions (Phase 0 headline).
    - ``val_ppl``: exp(val_loss).
    - ``val_label_loss``: mean CE on TRUE/FALSE positions only (Phase 0.5 headline).
    - ``val_label_acc``: argmax accuracy on TRUE/FALSE positions
        restricted to {TRUE_TOKEN_ID, FALSE_TOKEN_ID} — the actual
        binary-classification metric, ignoring the rest of the vocab.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_label_nll = 0.0
    total_label_tokens = 0
    total_label_correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            output = model(input_ids, mask=mask)
            logits = output.logits

            B, L, V = logits.shape
            flat_logits = logits.reshape(B * L, V)
            flat_labels = labels.reshape(B * L)

            per_pos = nn.functional.cross_entropy(
                flat_logits, flat_labels,
                ignore_index=-100, reduction="none",
            )
            valid = flat_labels != -100
            is_label = (flat_labels == TRUE_TOKEN_ID) | (flat_labels == FALSE_TOKEN_ID)

            nll_sum = (per_pos * valid.float()).sum()
            label_nll_sum = (per_pos * is_label.float()).sum()
            label_count = int(is_label.sum().item())

            # Binary accuracy restricted to {TRUE, FALSE} logits at label positions.
            if label_count > 0:
                label_idx = is_label.nonzero(as_tuple=True)[0]
                label_logits = flat_logits[label_idx]  # (n_label, V)
                bin_logits = torch.stack(
                    [label_logits[:, FALSE_TOKEN_ID], label_logits[:, TRUE_TOKEN_ID]],
                    dim=-1,
                )  # (n_label, 2)
                bin_pred = bin_logits.argmax(dim=-1)  # 0 → FALSE, 1 → TRUE
                bin_target = (flat_labels[label_idx] == TRUE_TOKEN_ID).long()
                total_label_correct += int((bin_pred == bin_target).sum().item())

        total_nll += float(nll_sum.item())
        total_tokens += int(valid.sum().item())
        total_label_nll += float(label_nll_sum.item())
        total_label_tokens += label_count

    val_loss = total_nll / max(total_tokens, 1)
    out = {
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20)),
        "val_tokens": total_tokens,
    }
    if total_label_tokens > 0:
        out["val_label_loss"] = total_label_nll / total_label_tokens
        out["val_label_ppl"] = math.exp(min(out["val_label_loss"], 20))
        out["val_label_acc"] = total_label_correct / total_label_tokens
        out["val_label_tokens"] = total_label_tokens
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", default=None,
                   help="Phase 0 byte-level corpus from tapas_proto_to_aegir.py")
    p.add_argument("--val-parquet", default=None,
                   help="Phase 0 held-out parquet")
    p.add_argument("--phase05-synth-sql", default=None,
                   help="Phase 0.5: synth-SQL parquet from generate_synth_sql.py")
    p.add_argument("--phase05-counterfactual", default=None,
                   help="Phase 0.5: counterfactual parquet")
    p.add_argument("--phase05-val-frac", type=float, default=0.02,
                   help="Phase 0.5: held-out fraction for val (deterministic seed-shuffle)")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--model-size", type=str, default="small",
                   choices=["tiny", "small", "base"])
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lambda-lb", type=float, default=0.1)
    p.add_argument("--downsample-factor", type=float, default=2.0)
    p.add_argument("--label-token-weight", type=float, default=1.0,
                   help="Multiplier on per-position CE loss for positions "
                        "whose target is TRUE_TOKEN_ID or FALSE_TOKEN_ID. "
                        "Phase 0 (no labels): leave at 1.0. Phase 0.5: set "
                        "to ~100-500 so the binary-classification objective "
                        "actually drives gradients — otherwise a ~1020-byte "
                        "context vs single-label-token target dilutes the "
                        "Phase 0.5 signal into the byte-LM noise floor "
                        "(<0.1%% of gradient on the actual task).")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=4649)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs/pretrain")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="If >0, subsample the train set for debugging.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a state_dict checkpoint (e.g. a Phase 0 "
                        "best_model.pt) to load before training starts. "
                        "Used to chain Phase 0 → Phase 0.5 without "
                        "discarding the backbone's learning.")
    return p.parse_args()


def _git_sha() -> tuple[str, str, bool]:
    try:
        full = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO, text=True).strip()
        short = full[:10]
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=REPO, text=True).strip())
        return short, full, dirty
    except Exception:
        return "unknown", "unknown", False


def main() -> int:
    args = parse_args()

    # DDP setup
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if is_distributed:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main:
        print("=== Aegir Pretraining ===")
        print(f"Model size: {args.model_size}")
        print(f"Train: {args.train_parquet}")
        print(f"Val:   {args.val_parquet}")
        print(f"Distributed: {is_distributed}  AMP: {args.amp}")

    # Datasets — either Phase 0 (byte LM) or Phase 0.5 (table+text+label)
    if args.phase05_synth_sql or args.phase05_counterfactual:
        from aegir.data.phase05 import Phase05Dataset

        full = Phase05Dataset(
            synth_sql_parquet=args.phase05_synth_sql,
            counterfactual_parquet=args.phase05_counterfactual,
            max_length=args.max_length,
        )
        n_total = len(full)
        n_val = max(1, int(n_total * args.phase05_val_frac))
        rng_np = np.random.default_rng(args.seed)
        perm = rng_np.permutation(n_total)
        val_idx = sorted(perm[:n_val].tolist())
        train_idx = sorted(perm[n_val:].tolist())

        class _Subset(torch.utils.data.Dataset):
            def __init__(self, base, idx):
                self.base = base
                self.idx = idx

            def __len__(self):
                return len(self.idx)

            def __getitem__(self, i):
                return self.base[self.idx[i]]

        train_ds = _Subset(full, train_idx)
        val_ds = _Subset(full, val_idx)
        if is_main:
            print(f"Phase 0.5 dataset: train={len(train_ds):,}  val={len(val_ds):,}")
        rank_shards_disjoint = False  # Phase 0.5 holds full dataset per rank
    else:
        if not args.train_parquet or not args.val_parquet:
            raise SystemExit(
                "Either --train-parquet/--val-parquet (Phase 0) or "
                "--phase05-* (Phase 0.5) sources must be provided."
            )
        # Rank-shard the parquet so each rank holds only 1/world_size of
        # the byte_ids buffer. Combined with lazy as_py() row access,
        # peak per-rank memory drops from ~14 GB (full table) to
        # ~2.3 GB (1/6 shard).
        train_ds = ParquetByteDataset(
            args.train_parquet, max_length=args.max_length,
            rank=rank, world_size=world_size,
        )
        val_ds = ParquetByteDataset(
            args.val_parquet, max_length=args.max_length,
            rank=rank, world_size=world_size,
        )
        if args.max_train_samples and args.max_train_samples < len(train_ds):
            full_ds = train_ds

            class _Subset(torch.utils.data.Dataset):
                def __init__(self, base, n):
                    self.base = base
                    self.n = n

                def __len__(self):
                    return self.n

                def __getitem__(self, i):
                    return self.base[i]

            train_ds = _Subset(full_ds, args.max_train_samples)
        if is_main:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"Phase 0 dataset (rank-sharded): "
                  f"train_local={len(train_ds):,}  "
                  f"val_local={len(val_ds):,}  rank0_rss={rss_mb:.0f} MB",
                  flush=True)
        # Each rank already holds a disjoint shard, so we use
        # RandomSampler (not DistributedSampler) for Phase 0.
        rank_shards_disjoint = True

    if rank_shards_disjoint:
        train_sampler = RandomSampler(train_ds)
        val_sampler = None
    elif is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = RandomSampler(train_ds)
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=lm_collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=lm_collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # Model + optim
    model = make_pretrain_model(args).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params:,}")

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise SystemExit(f"--resume-from path not found: {ckpt_path}")
        if is_main:
            print(f"Loading state_dict from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Allow shape mismatches on the LM head (e.g., vocab change) by
        # surfacing strict-load failures with the offending keys.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            if missing:
                print(f"  missing keys: {missing[:10]}"
                      + (f" ... (+{len(missing)-10} more)" if len(missing) > 10 else ""))
            if unexpected:
                print(f"  unexpected keys: {unexpected[:10]}"
                      + (f" ... (+{len(unexpected)-10} more)" if len(unexpected) > 10 else ""))
            print(f"  loaded {n_params - sum(p.numel() for n, p in model.named_parameters() if n in missing):,} of {n_params:,} params")

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
    raw_model = getattr(model, "module", model)

    param_groups = group_params(raw_model)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 0.1) + 0.1  # decay to 10%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Output dir
    git_short, git_full, git_dirty = _git_sha()
    run_id = (
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{git_short}_{args.model_size}_pretrain"
    )
    run_dir = Path(args.output_dir) / "runs" / run_id
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metadata.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "model_size": args.model_size,
                "seed": args.seed,
                "vocab_size": args.vocab_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "downsample_factor": args.downsample_factor,
                "lambda_lb": args.lambda_lb,
                "label_token_weight": args.label_token_weight,
                "amp": args.amp,
                "git_short_sha": git_short,
                "git_full_sha": git_full,
                "git_dirty": git_dirty,
                "utc_start": datetime.now(tz=timezone.utc).isoformat(),
                "argv": sys.argv,
                "host": os.uname().nodename,
                "python": sys.version.split()[0],
                "num_params": sum(p.numel() for p in raw_model.parameters()),
            }, f, indent=2)
        print(f"Run dir: {run_dir}")

    # Training
    metrics_log: list[dict] = []
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, args,
            epoch_idx=epoch + 1, is_main=is_main,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, args)

        if is_main:
            train_label_str = (
                f" tlabel={train_metrics['train_label_loss']:.4f}"
                if train_metrics.get('train_label_loss') is not None else ""
            )
            val_label_str = (
                f" vlabel={val_metrics['val_label_loss']:.4f} "
                f"vlabel_acc={val_metrics['val_label_acc']:.4f}"
                if 'val_label_loss' in val_metrics else ""
            )
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"train loss={train_metrics['train_loss']:.4f} "
                f"(task={train_metrics['train_task_loss']:.4f} "
                f"lb={train_metrics['train_lb_loss']:.4f}){train_label_str} | "
                f"val loss={val_metrics['val_loss']:.4f} "
                f"ppl={val_metrics['val_ppl']:.2f}{val_label_str} | "
                f"{train_metrics['wall_seconds']:.1f}s",
                flush=True,
            )

            metrics_log.append({
                "epoch": epoch + 1,
                **{k: v for k, v in train_metrics.items() if k != "boundary"},
                **val_metrics,
                "boundary": train_metrics["boundary"],
            })
            with open(run_dir / "metrics.json", "w") as f:
                json.dump({"epochs": metrics_log, "final": None}, f, indent=2)

            # Checkpoint on improvement. For Phase 0.5 (label_token_weight > 1
            # OR val_label_loss present), use label loss as the gate so we
            # don't keep checkpoints that overfit byte reconstruction at the
            # expense of label accuracy.
            if 'val_label_loss' in val_metrics:
                gate = val_metrics['val_label_loss']
                gate_name = 'val_label_loss'
            else:
                gate = val_metrics['val_loss']
                gate_name = 'val_loss'
            if gate < best_val_loss:
                best_val_loss = gate
                ckpt_path = run_dir / "best_model.pt"
                torch.save(raw_model.state_dict(), ckpt_path)
                print(f"  saved best_model.pt ({gate_name}={best_val_loss:.4f})")

        if is_distributed:
            dist.barrier()

    if is_main:
        # Final checkpoint regardless of val.
        torch.save(raw_model.state_dict(), run_dir / "final_model.pt")
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({
                "epochs": metrics_log,
                "final": {
                    "best_val_loss": best_val_loss,
                    "last_epoch": args.epochs,
                    "num_epochs_completed": args.epochs,
                },
            }, f, indent=2)
        print(f"\nDone. Best val loss = {best_val_loss:.4f}, "
              f"ppl = {math.exp(min(best_val_loss, 20)):.2f}")

    if is_distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
