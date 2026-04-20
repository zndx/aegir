#!/usr/bin/env python3
"""Evaluate an Aegir checkpoint in bits/byte on FineWeb-Edu tail.

Counterpart to ``eval_rwkv7_baseline.py``. Uses the exact same eval
byte window so the two numbers are directly comparable. Aegir is
byte-level natively, so no tokenizer conversion needed — the NLL in
nats is already per-byte.

Usage:
    uv run --no-sync python scripts/eval_aegir_baseline.py \
        --checkpoint outputs/repro/20260420T*/final.pt \
        --n-layer 12 --d-model 768 --max-bytes 1_000_000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForCausalLM


log = logging.getLogger("eval_aegir")


def _repro_config(vocab_size=260, n_layer=12, d_model=768):
    return AegirConfig(
        arch_layout=[f"w{n_layer}"],
        d_model=[d_model],
        d_intermediate=[0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(num_heads=[d_model // 64], rotary_emb_dim=[64], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1, task_type="cta",
    )


def _read_eval_bytes(data_dir: Path, max_bytes: int, skip_bytes: int) -> bytes:
    shards = sorted(data_dir.glob("*.txt"))
    if not shards:
        raise FileNotFoundError(f"No .txt shards in {data_dir}")
    last = shards[-1]
    size = last.stat().st_size
    start = max(0, size - max_bytes - skip_bytes)
    log.info("Eval window: last %d bytes of %s (file size %.2f GB)",
             max_bytes, last.name, size / 2**30)
    with open(last, "rb") as fh:
        fh.seek(start)
        return fh.read(max_bytes)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/raid/datasets/fineweb-edu"))
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    ap.add_argument("--skip-bytes", type=int, default=0)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--vocab-size", type=int, default=260)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    log.info("Building Aegir (L=%d D=%d vocab=%d)...",
             args.n_layer, args.d_model, args.vocab_size)
    config = _repro_config(
        vocab_size=args.vocab_size, n_layer=args.n_layer, d_model=args.d_model,
    )
    model = AegirForCausalLM(config=config, device=device, dtype=dtype)
    log.info("Loading checkpoint: %s", args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info("load: %d missing, %d unexpected", len(missing), len(unexpected))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params: %s", f"{n_params:,}")

    raw = _read_eval_bytes(args.data_dir, args.max_bytes, args.skip_bytes)
    log.info("eval bytes: %d (%.2f MB)", len(raw), len(raw) / 2**20)

    # Aegir is byte-level; tokens == bytes. Score at sliding half-overlap
    # windows, taking loss on last half of each non-first window (same
    # protocol as eval_rwkv7_baseline).
    ctx = args.ctx_len
    stride = ctx // 2
    byte_ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()

    total_nll_nats = 0.0
    total_scored_bytes = 0
    t0 = time.time()
    n_windows = 0
    with torch.no_grad():
        for start in range(0, max(1, byte_ids.shape[0] - ctx - 1), stride):
            inp = byte_ids[start : start + ctx].to(device)
            tgt = byte_ids[start + 1 : start + ctx + 1].to(device)
            if tgt.shape[0] < ctx:
                break
            score_start = 0 if start == 0 else ctx // 2
            inp_b = inp[None, :]
            mask = torch.ones(1, ctx, dtype=torch.bool, device=device)
            out = model(input_ids=inp_b, mask=mask)
            logp = F.log_softmax(out.logits[0].float(), dim=-1)
            nll = -logp.gather(1, tgt[:, None]).squeeze(1)
            scored = nll[score_start:]
            total_nll_nats += scored.sum().item()
            total_scored_bytes += scored.shape[0]
            n_windows += 1
            if n_windows % 20 == 0:
                bpb = total_nll_nats / max(1, total_scored_bytes) / math.log(2)
                log.info("  win %3d  scored=%d  running bits/byte=%.4f",
                         n_windows, total_scored_bytes, bpb)

    elapsed = time.time() - t0
    bits_per_byte = total_nll_nats / max(1, total_scored_bytes) / math.log(2)

    log.info("=" * 60)
    log.info("FINAL: %d windows, %d scored bytes, %.1fs",
             n_windows, total_scored_bytes, elapsed)
    log.info("  nll/byte (nats): %.4f", total_nll_nats / max(1, total_scored_bytes))
    log.info("  BITS / BYTE    : %.4f", bits_per_byte)
    log.info("=" * 60)

    result = {
        "checkpoint": str(args.checkpoint),
        "n_layer": args.n_layer, "d_model": args.d_model,
        "params": n_params,
        "eval_bytes_total": len(raw),
        "scored_bytes": total_scored_bytes,
        "nll_per_byte_nats": total_nll_nats / max(1, total_scored_bytes),
        "bits_per_byte": bits_per_byte,
        "ctx_len": ctx, "stride": stride, "n_windows": n_windows,
        "eval_seconds": elapsed,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        log.info("Saved: %s", args.out)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
