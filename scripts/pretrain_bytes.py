#!/usr/bin/env python3
"""Byte-level pretraining for Aegir on raw GitTables (Stage B).

This is the "dense-supervision" half of the sparse→dense reframe
(see docs/current/training_regime.md). Reads GitTables parquets,
serializes each table to a byte stream, feeds Aegir its native
next-byte prediction objective via AegirForCausalLM.

Load-bearing question: does the architecture converge at all under
its designed training regime, or is there a deeper structural issue
exposed by pretraining (e.g., v_first collapse, DeChunk EMA stability)?

Budget for first probe: ~100M bytes. On `small` config with batch 8
and seq_len 1024 that's ~12k steps — enough to see whether loss is
actually decreasing vs plateau'd on entropy floor.

Usage:
    uv run --no-sync python scripts/pretrain_bytes.py \
        --gittables-dir /raid/datasets/gittables \
        --model-size small --seq-len 1024 --batch-size 8 \
        --max-bytes 100_000_000 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForCausalLM
from aegir.utils.train import group_params

log = logging.getLogger("pretrain_bytes")


# ── Byte-stream dataset ───────────────────────────────────────


class GitTablesByteStream(IterableDataset):
    """Streams fixed-length byte sequences from GitTables parquets.

    Each parquet → concatenate each column's values as UTF-8 strings
    separated by '\\x01' (cell delimiter) and '\\x02' (column delimiter).
    Emit fixed-length `seq_len`-byte blocks with `\\x03` (table delimiter)
    separating tables. No label, no annotations — raw byte stream.

    Args:
        data_dir: Path with *.parquet files.
        seq_len: bytes per training sample.
        max_bytes: stop after emitting this many total bytes
            (0 = unlimited). Budget control for probe runs.
        shuffle_files: if True, randomize file order per epoch.
        seed: RNG seed for reproducible shuffling.
    """

    CELL_DELIM = b"\x01"
    COL_DELIM = b"\x02"
    TABLE_DELIM = b"\x03"

    def __init__(
        self,
        data_dir: Path,
        seq_len: int = 1024,
        max_bytes: int = 100_000_000,
        shuffle_files: bool = True,
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.max_bytes = max_bytes
        self.shuffle_files = shuffle_files
        self.seed = seed

    def _iter_table_bytes(self, pq_path: Path) -> bytes:
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(pq_path)
        except Exception:
            return b""
        parts: list[bytes] = []
        for col in table.column_names:
            for val in table.column(col).to_pylist()[:200]:
                if val is None:
                    continue
                parts.append(str(val).encode("utf-8", errors="replace"))
                parts.append(self.CELL_DELIM)
            parts.append(self.COL_DELIM)
        parts.append(self.TABLE_DELIM)
        return b"".join(parts)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(self.data_dir.rglob("*.parquet"))
        if self.shuffle_files:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        buf = bytearray()
        total = 0
        for pq_path in files:
            if self.max_bytes and total >= self.max_bytes:
                return
            buf.extend(self._iter_table_bytes(pq_path))
            while len(buf) >= self.seq_len + 1:
                chunk = bytes(buf[: self.seq_len + 1])
                del buf[: self.seq_len]
                # +1 byte → input_ids[0:seq_len], targets[1:seq_len+1]
                yield torch.frombuffer(chunk, dtype=torch.uint8).long()
                total += self.seq_len
                if self.max_bytes and total >= self.max_bytes:
                    return


def byte_collate(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collate list of (seq_len+1,) byte tensors into input/target batch."""
    stacked = torch.stack(batch)
    return {
        "input_ids": stacked[:, :-1],
        "targets": stacked[:, 1:],
        "mask": torch.ones(stacked.shape[0], stacked.shape[1] - 1, dtype=torch.bool),
    }


# ── Model config ───────────────────────────────────────────────


def _config(size: str, vocab_size: int = 260) -> AegirConfig:
    """Small/base configs matching train.py but with byte-sized vocab."""
    if size == "tiny":
        return AegirConfig(
            arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
            d_model=[128, 192, 192],
            d_intermediate=[0, 0, 0],
            vocab_size=vocab_size,
            ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
            attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[8, 12, 12], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=1,
            task_type="cta",
        )
    if size == "small":
        return AegirConfig(
            arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
            d_model=[256, 384, 384],
            d_intermediate=[0, 0, 0],
            vocab_size=vocab_size,
            ssm_cfg=SSMConfig(),
            attn_cfg=AttnConfig(num_heads=[4, 6, 6], rotary_emb_dim=[16, 24, 24], window_size=[]),
            rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
            num_labels=1,
            task_type="cta",
        )
    raise ValueError(f"unknown size {size!r}")


# ── Training loop ──────────────────────────────────────────────


def _cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    import math

    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gittables-dir", type=Path, default=Path("/raid/datasets/gittables"))
    ap.add_argument("--model-size", default="small", choices=["tiny", "small"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch × grad_accum)")
    ap.add_argument("--max-bytes", type=int, default=100_000_000,
                    help="Total byte budget across the run (0 = unlimited)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/pretrain"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    out_dir = args.out_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    # Data
    ds = GitTablesByteStream(
        data_dir=args.gittables_dir,
        seq_len=args.seq_len,
        max_bytes=args.max_bytes,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=byte_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    config = _config(args.model_size, vocab_size=260)
    model = AegirForCausalLM(config=config, device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params: %s  config: %s", f"{n_params:,}", args.model_size)

    # Optimizer
    param_groups = group_params(model)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )

    # Estimate total optimizer steps from byte budget
    bytes_per_batch = args.batch_size * args.seq_len
    bytes_per_step = bytes_per_batch * args.grad_accum
    total_steps = max(
        100,
        (args.max_bytes // bytes_per_step) if args.max_bytes else 10_000,
    )
    log.info(
        "bytes/step=%d total_steps=%d warmup=%d",
        bytes_per_step, total_steps, args.warmup_steps,
    )

    # Save metadata
    (out_dir / "metadata.json").write_text(json.dumps({
        "model_size": args.model_size,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_bytes": args.max_bytes,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "num_params": n_params,
        "total_steps_planned": total_steps,
        "vocab_size": 260,
        "cmdline": sys.argv,
    }, indent=2))

    metrics_path = out_dir / "metrics.jsonl"

    # Training
    model.train()
    step = 0
    micro_step = 0
    optimizer.zero_grad()
    loss_acc = 0.0
    samples_acc = 0
    t_last = time.time()
    scaler = None  # bf16 doesn't need grad scaler

    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, mask=mask)
            logits = out.logits  # (B, L, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                reduction="mean",
            )
            (loss / args.grad_accum).backward()
            loss_acc += loss.item()
            samples_acc += 1

            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr_scale = _cosine_warmup(step, args.warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                mean_loss = loss_acc / max(1, samples_acc)
                elapsed = time.time() - t_last
                throughput = (args.log_every * bytes_per_step) / max(elapsed, 1e-3)
                log.info(
                    "step %6d  loss %.4f  lr %.2e  %.0f bytes/s",
                    step, mean_loss, args.lr * lr_scale, throughput,
                )
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": mean_loss,
                        "lr": args.lr * lr_scale,
                        "bytes_per_sec": throughput,
                        "peak_cuda_mem_mb": (
                            torch.cuda.max_memory_allocated() / 2**20
                            if torch.cuda.is_available() else 0
                        ),
                    }) + "\n")
                loss_acc = 0.0
                samples_acc = 0
                t_last = time.time()

            if step > 0 and step % args.ckpt_every == 0:
                torch.save(model.state_dict(), out_dir / f"ckpt_{step:06d}.pt")
                log.info("saved checkpoint at step %d", step)

            if step >= total_steps:
                break
    except KeyboardInterrupt:
        log.info("interrupted by user")

    torch.save(model.state_dict(), out_dir / "final.pt")
    log.info("saved final.pt; training complete at step %d", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
