#!/usr/bin/env python3
"""Mixed-corpus pretraining over MIXTURE.json.

Reads the manifest produced by scripts/compose_corpus_manifest.py and
samples byte windows from the declared slices, weighted by the
normalised weight per slice. Hypothesis: domain-mixed pretraining
reduces bits/byte on a held-out SQL corpus without sacrificing
FineWeb-Edu perplexity.

Based on scripts/pretrain_fineweb.py; the only structural change is
the dataset: instead of a single directory, it takes a manifest of
per-slice file lists + weights.

Usage:
    uv run --no-sync python scripts/pretrain_mixed.py \
        --manifest /raid/datasets/aegir-corpus-v1/MIXTURE.json \
        --max-bytes 1_000_000_000 --seq-len 512
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

log = logging.getLogger("pretrain_mixed")


class MixedCorpusStream(IterableDataset):
    """Weighted streaming sampler over corpus slices.

    For each chunk, picks a slice by weight, fills an internal byte
    buffer for that slice from its file list (round-robin with
    per-epoch shuffle), then emits a seq_len+1 byte window.

    Slice internal state is kept per-worker so DataLoader workers
    don't contend.
    """

    def __init__(
        self,
        slices: list[dict],
        seq_len: int = 512,
        max_bytes: int = 1_000_000_000,
        seed: int = 4649,
    ):
        self.slices = slices
        self.seq_len = seq_len
        self.max_bytes = max_bytes
        self.seed = seed
        tot = sum(s["weight_normalised"] for s in slices)
        self.probs = [s["weight_normalised"] / tot for s in slices]

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id * 997)

        state: list[dict] = []
        for slc in self.slices:
            files = list(slc["files"])
            # Partition file list across workers
            files = files[worker_id::num_workers]
            rng.shuffle(files)
            state.append({
                "name": slc["name"],
                "files": files,
                "file_idx": 0,
                "buf": bytearray(),
                "fh": None,
            })

        total = 0
        CHUNK = 1 << 20  # 1 MB read per file chunk
        indices = list(range(len(self.slices)))
        while total < self.max_bytes:
            i = rng.choices(indices, weights=self.probs, k=1)[0]
            s = state[i]
            # Make sure buffer has enough for one sequence
            while len(s["buf"]) < self.seq_len + 1:
                if s["fh"] is None:
                    if not s["files"]:
                        break
                    fpath = s["files"][s["file_idx"] % len(s["files"])]
                    s["file_idx"] += 1
                    try:
                        s["fh"] = open(fpath, "rb")
                    except FileNotFoundError:
                        s["fh"] = None
                        continue
                block = s["fh"].read(CHUNK)
                if not block:
                    s["fh"].close()
                    s["fh"] = None
                    if s["file_idx"] >= len(s["files"]):
                        rng.shuffle(s["files"])
                        s["file_idx"] = 0
                    continue
                s["buf"].extend(block)
            if len(s["buf"]) < self.seq_len + 1:
                continue
            chunk = bytes(s["buf"][: self.seq_len + 1])
            del s["buf"][: self.seq_len]
            yield torch.frombuffer(chunk, dtype=torch.uint8).long()
            total += self.seq_len


def byte_collate(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = torch.stack(batch)
    return {
        "input_ids": stacked[:, :-1],
        "targets": stacked[:, 1:],
        "mask": torch.ones(stacked.shape[0], stacked.shape[1] - 1, dtype=torch.bool),
    }


def _repro_config(vocab_size: int = 260, n_layer: int = 12, d_model: int = 768) -> AegirConfig:
    return AegirConfig(
        arch_layout=[f"w{n_layer}"],
        d_model=[d_model],
        d_intermediate=[0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(
            num_heads=[d_model // 64], rotary_emb_dim=[64], window_size=[],
        ),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1, task_type="cta",
    )


def _cosine_lr(step: int, warmup: int, total: int, lr_init: float, lr_final: float) -> float:
    if step < warmup:
        return lr_init * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--vocab-size", type=int, default=260)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-bytes", type=int, default=1_000_000_000)
    # BlinkDL hyperparameters (same as pretrain_fineweb.py)
    ap.add_argument("--lr-init", type=float, default=6e-4)
    ap.add_argument("--lr-final", type=float, default=6e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--adam-eps", type=float, default=1e-18)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/mixed"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    out_dir = args.out_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    # Load manifest
    manifest = json.loads(args.manifest.read_text())
    log.info("Manifest: %d slices, %.2f GB total",
             len(manifest["slices"]), manifest["total_bytes"] / 2**30)
    for s in manifest["slices"]:
        log.info("  %-22s  w=%.3f  files=%d  %.2f MB",
                 s["name"], s.get("weight_normalised", 0),
                 s["n_files"], s["bytes"] / 2**20)

    ds = MixedCorpusStream(
        slices=manifest["slices"],
        seq_len=args.seq_len,
        max_bytes=args.max_bytes,
        seed=args.seed,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, collate_fn=byte_collate,
        num_workers=args.num_workers, pin_memory=True,
    )

    config = _repro_config(
        vocab_size=args.vocab_size, n_layer=args.n_layer, d_model=args.d_model,
    )
    model = AegirForCausalLM(config=config, device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params: %s  arch: L=%d D=%d  seq=%d",
             f"{n_params:,}", args.n_layer, args.d_model, args.seq_len)

    from aegir.utils.train import group_params
    param_groups = group_params(model)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr_init,
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    bytes_per_step = args.batch_size * args.seq_len * args.grad_accum
    total_steps = max(100, args.max_bytes // bytes_per_step)
    log.info("bytes/step=%d total_steps=%d warmup=%d lr=%.0e→%.0e",
             bytes_per_step, total_steps, args.warmup_steps, args.lr_init, args.lr_final)

    (out_dir / "metadata.json").write_text(json.dumps({
        "arch_layout": config.arch_layout,
        "d_model": config.d_model,
        "vocab_size": args.vocab_size,
        "n_layer": args.n_layer,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_bytes": args.max_bytes,
        "lr_init": args.lr_init, "lr_final": args.lr_final,
        "warmup_steps": args.warmup_steps,
        "betas": [args.beta1, args.beta2], "adam_eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_params": n_params,
        "total_steps_planned": total_steps,
        "manifest": str(args.manifest),
        "cmdline": sys.argv,
    }, indent=2))

    metrics_path = out_dir / "metrics.jsonl"
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
                targets.reshape(-1), reduction="mean",
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
                log.info("step %6d  loss %.4f  lr %.3e  %.0f bytes/s",
                         step, mean_loss, lr, throughput)
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
    log.info("saved final.pt at step %d", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
