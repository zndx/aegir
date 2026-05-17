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

from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForCausalLM
from aegir.utils.train import (
    boundary_diagnostics,
    group_params,
    load_balancing_loss,
)


# ─────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────

class ParquetByteDataset(Dataset):
    """Loads a parquet of (table_id, byte_ids, length) rows in memory."""

    def __init__(self, path: str | Path, max_length: int):
        table = pq.read_table(str(path), columns=["byte_ids", "length"])
        self.byte_ids = table["byte_ids"].to_pylist()
        self.lengths = table["length"].to_numpy()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.byte_ids)

    def __getitem__(self, idx):
        ids = self.byte_ids[idx][: self.max_length]
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
    """One pretrain epoch — next-token CE + load-balancing loss."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_lb_loss = 0.0
    n_batches = 0
    n_tokens = 0
    boundary_accum: dict[str, float] = {}

    log_interval = getattr(args, "log_interval", 20) or 20
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

            B, L, V = logits.shape
            task_loss = loss_fn(logits.reshape(B * L, V), labels.reshape(B * L))

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
            print(
                f"  [e{epoch_idx} step {step_idx+1:6d}]  "
                f"loss={avg_loss:.4f}  ppl={math.exp(min(avg_loss, 20)):.2f}  "
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
        "train_lb_loss": total_lb_loss / max(n_batches, 1),
        "boundary": _finalize(boundary_accum, n_batches),
        "n_tokens": n_tokens,
        "wall_seconds": time.time() - epoch_start,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, args):
    """Held-out perplexity."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            output = model(input_ids, mask=mask)
            logits = output.logits
            B, L, V = logits.shape
            # Use sum reduction so per-token NLL aggregates correctly across batches.
            nll = nn.functional.cross_entropy(
                logits.reshape(B * L, V),
                labels.reshape(B * L),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = int((labels != -100).sum().item())

        total_nll += float(nll.item())
        total_tokens += n_valid

    val_loss = total_nll / max(total_tokens, 1)
    return {"val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20)), "val_tokens": total_tokens}


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
    else:
        if not args.train_parquet or not args.val_parquet:
            raise SystemExit(
                "Either --train-parquet/--val-parquet (Phase 0) or "
                "--phase05-* (Phase 0.5) sources must be provided."
            )
        train_ds = ParquetByteDataset(args.train_parquet, max_length=args.max_length)
        val_ds = ParquetByteDataset(args.val_parquet, max_length=args.max_length)
        if args.max_train_samples and args.max_train_samples < len(train_ds):
            train_ds.byte_ids = train_ds.byte_ids[: args.max_train_samples]
            train_ds.lengths = train_ds.lengths[: args.max_train_samples]
        if is_main:
            print(f"Phase 0 dataset: train={len(train_ds):,}  val={len(val_ds):,}")

    if is_distributed:
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
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, args,
            epoch_idx=epoch + 1, is_main=is_main,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, args)

        if is_main:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"train loss={train_metrics['train_loss']:.4f} "
                f"(task={train_metrics['train_task_loss']:.4f} "
                f"lb={train_metrics['train_lb_loss']:.4f}) | "
                f"val loss={val_metrics['val_loss']:.4f} ppl={val_metrics['val_ppl']:.2f} | "
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

            # Checkpoint on improvement.
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                ckpt_path = run_dir / "best_model.pt"
                torch.save(raw_model.state_dict(), ckpt_path)
                print(f"  saved best_model.pt (val_loss={best_val_loss:.4f})")

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
