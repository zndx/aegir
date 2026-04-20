#!/usr/bin/env python3
"""Reproduction pretraining: BlinkDL RWKV-7 recipe on FineWeb-Edu.

Conservative architecture-isolation test. Everything matches
``~/local/src/oss/rwkv-lm/RWKV-v7/train_temp/demo-training-run.sh``
down to the optimizer ``eps=1e-18`` except:
  - Our byte-level tokenizer (vocab 260) replaces the 65536 rwkv-vocab.
  - Flat arch (``["w12"]``) — no H-Net chunking, to match BlinkDL's
    12-layer uniform stack exactly. The next experiment reinstates
    hierarchy and measures the delta.

The point: if this reproduces the expected loss curve, the architecture
+ fla kernels + our training pipeline are sound at the RWKV-7 level,
and the failure modes we've seen in Stages A/C are attributable to the
data/task coupling, not the model.

Reference hyperparameters (BlinkDL demo):

    lr_init      = 6e-4
    lr_final     = 6e-5     (cosine decay)
    warmup_steps = 10       (yes, ten — not 500)
    betas        = (0.9, 0.99)
    adam_eps     = 1e-18
    weight_decay = 0.001
    head_size    = 64
    ctx_len      = 512
    bsz          = 16
    precision    = bf16

Usage:
    uv run --no-sync python scripts/pretrain_fineweb.py \
        --data-dir /raid/datasets/fineweb-edu --max-bytes 300_000_000

Output at outputs/repro/{run_id}/:
    metadata.json
    metrics.jsonl (per log_every step)
    final.pt
    ckpt_NNNNNN.pt (every ckpt_every)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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


log = logging.getLogger("pretrain_fineweb")


# ── Corpus ──────────────────────────────────────────────────────


class FineWebByteStream(IterableDataset):
    """Stream fixed-length byte sequences from FineWeb-Edu *.txt shards.

    Each ``.txt`` shard is UTF-8 text with document boundaries marked
    by ``\\x03``. We just read the raw bytes and chunk them to seq_len;
    document boundaries are preserved as literal bytes in the stream,
    which is the correct signal for a byte-level LM (H-Net's chunker
    would learn to treat them as boundary hints, a flat RWKV just
    sees a token like any other).
    """

    def __init__(
        self,
        data_dir: Path,
        seq_len: int = 512,
        max_bytes: int = 300_000_000,
        shuffle_files: bool = True,
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.max_bytes = max_bytes
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(self.data_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(
                f"No *.txt shards under {self.data_dir}. "
                f"Run `scripts/download_fineweb.py` first."
            )
        if self.shuffle_files:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        buf = bytearray()
        total = 0
        CHUNK = 1 << 20  # 1 MB read
        for f in files:
            if self.max_bytes and total >= self.max_bytes:
                return
            with open(f, "rb") as fh:
                while True:
                    block = fh.read(CHUNK)
                    if not block:
                        break
                    buf.extend(block)
                    while len(buf) >= self.seq_len + 1:
                        chunk = bytes(buf[: self.seq_len + 1])
                        del buf[: self.seq_len]
                        yield torch.frombuffer(chunk, dtype=torch.uint8).long()
                        total += self.seq_len
                        if self.max_bytes and total >= self.max_bytes:
                            return


def byte_collate(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = torch.stack(batch)
    return {
        "input_ids": stacked[:, :-1],
        "targets": stacked[:, 1:],
        "mask": torch.ones(stacked.shape[0], stacked.shape[1] - 1, dtype=torch.bool),
    }


# ── Model config — flat L12 D768 (strict BlinkDL repro) ─────────


def _repro_config(vocab_size: int = 260, n_layer: int = 12, d_model: int = 768) -> AegirConfig:
    return AegirConfig(
        # Flat arch: single innermost stage with n_layer RWKV-7 blocks.
        # No RoutingModule, no chunking — the H-Net hierarchy is deliberately
        # off for this conservative reproduction.
        arch_layout=[f"w{n_layer}"],
        d_model=[d_model],
        d_intermediate=[0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(
            num_heads=[d_model // 64], rotary_emb_dim=[64], window_size=[],
        ),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1,
        task_type="cta",
    )


# ── Training ────────────────────────────────────────────────────


def _cosine_lr(step: int, warmup: int, total: int,
               lr_init: float, lr_final: float) -> float:
    """BlinkDL-style: 10-step linear warmup to lr_init, then cosine to lr_final."""
    if step < warmup:
        return lr_init * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("/raid/datasets/fineweb-edu"))
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--vocab-size", type=int, default=260)
    ap.add_argument("--seq-len", type=int, default=512, help="BlinkDL default: 512")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="Effective batch = batch × grad_accum")
    ap.add_argument("--max-bytes", type=int, default=300_000_000,
                    help="Total byte budget (0 = unlimited)")
    # BlinkDL exact hyperparameters
    ap.add_argument("--lr-init", type=float, default=6e-4)
    ap.add_argument("--lr-final", type=float, default=6e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--adam-eps", type=float, default=1e-18)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # bookkeeping
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/repro"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    out_dir = args.out_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    # Data
    ds = FineWebByteStream(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        max_bytes=args.max_bytes,
        seed=args.seed,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, collate_fn=byte_collate,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    config = _repro_config(
        vocab_size=args.vocab_size, n_layer=args.n_layer, d_model=args.d_model,
    )
    model = AegirForCausalLM(config=config, device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "params: %s  arch: L=%d D=%d vocab=%d  seq=%d",
        f"{n_params:,}", args.n_layer, args.d_model, args.vocab_size, args.seq_len,
    )

    # Optimizer — BlinkDL recipe exactly: betas, eps, wd
    # (RWKV-LM typically excludes 1D params / biases / norms from weight decay.
    #  Our `group_params` does this automatically via the _optim annotations.)
    from aegir.utils.train import group_params

    param_groups = group_params(model)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr_init,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Estimate total steps from byte budget
    bytes_per_batch = args.batch_size * args.seq_len
    bytes_per_step = bytes_per_batch * args.grad_accum
    total_steps = max(100, (args.max_bytes // bytes_per_step) if args.max_bytes else 100_000)
    log.info(
        "bytes/step=%d total_steps=%d warmup=%d lr=%.0e→%.0e",
        bytes_per_step, total_steps, args.warmup_steps, args.lr_init, args.lr_final,
    )

    (out_dir / "metadata.json").write_text(json.dumps({
        "arch_layout": config.arch_layout,
        "d_model": config.d_model,
        "vocab_size": args.vocab_size,
        "n_layer": args.n_layer,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_bytes": args.max_bytes,
        "lr_init": args.lr_init,
        "lr_final": args.lr_final,
        "warmup_steps": args.warmup_steps,
        "betas": [args.beta1, args.beta2],
        "adam_eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_params": n_params,
        "total_steps_planned": total_steps,
        "recipe_source": "~/local/src/oss/rwkv-lm/RWKV-v7/train_temp/demo-training-run.sh",
        "cmdline": sys.argv,
    }, indent=2))

    metrics_path = out_dir / "metrics.jsonl"

    # Training loop
    model.train()
    step = 0
    micro_step = 0
    optimizer.zero_grad()
    loss_acc = 0.0
    samples_acc = 0
    t_last = time.time()

    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, mask=mask)
            loss = F.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)).float(),
                targets.reshape(-1),
                reduction="mean",
            )
            (loss / args.grad_accum).backward()
            loss_acc += loss.item()
            samples_acc += 1

            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr = _cosine_lr(step, args.warmup_steps, total_steps, args.lr_init, args.lr_final)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                mean_loss = loss_acc / max(1, samples_acc)
                elapsed = time.time() - t_last
                throughput = (args.log_every * bytes_per_step) / max(elapsed, 1e-3)
                log.info(
                    "step %6d  loss %.4f  lr %.3e  %.0f bytes/s",
                    step, mean_loss, lr, throughput,
                )
                with open(metrics_path, "a") as fh:
                    fh.write(json.dumps({
                        "step": step, "loss": mean_loss, "lr": lr,
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
    log.info("saved final.pt; complete at step %d", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
