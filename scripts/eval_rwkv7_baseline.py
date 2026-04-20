#!/usr/bin/env python3
"""Evaluate an official BlinkDL RWKV-7 checkpoint on our FineWeb-Edu bytes.

Produces a ``bits/byte`` number — tokenizer-agnostic perplexity measure —
as a calibration anchor for our own training runs. Aegir's training loss
is in nats/byte (native byte-level); this script puts BlinkDL's BPE-level
loss on the same scale by dividing per-token NLL nats by the number of
UTF-8 bytes each token decodes to.

Model definition is *verbatim* from
``~/local/src/oss/rwkv-lm/RWKV-v7/rwkv_v7_demo.py``, with one substitution:
the ``RWKV7_OP`` wkv7 operator is replaced by ``fla.ops.rwkv7.chunk_rwkv7``
so we don't need to compile BlinkDL's CUDA kernel. fla's kernel is
numerically equivalent and already operational in our training stack.

Config is auto-detected from the state dict at load time (n_layer,
n_embd, LoRA dims, FFN dim, vocab), so the same script runs on
0.1B / 0.4B / 1.5B / 2.9B / 7.2B / 13.3B checkpoints.

Usage:
    uv run --no-sync python scripts/eval_rwkv7_baseline.py \
        --checkpoint /raid/datasets/rwkv-7-world/rwkv7-g1e-7.2b-*.pth \
        --data-dir /raid/datasets/fineweb-edu \
        --max-bytes 1_000_000 --ctx-len 2048
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.rwkv7.chunk import chunk_rwkv7

log = logging.getLogger("eval_rwkv7")


# ── BlinkDL RWKV-7 tokenizer (verbatim) ─────────────────────────


class RWKV_TOKENIZER:
    table: list
    good: list
    wlen: list

    def __init__(self, file_name):
        self.idx2token = {}
        sorted_list = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            x = eval(line[line.index(" ") : line.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(" "):])
            sorted_list += [x]
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_list))):
            s = sorted_list[i]
            if len(s) >= 2:
                s0, s1 = int(s[0]), int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i : i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens


# ── fla-backed WKV7 wrapper ─────────────────────────────────────


def RWKV7_OP(r, w, k, v, a, b, head_size: int):
    """fla-backed wkv7 in place of BlinkDL's CUDA kernel.

    Inputs are (B, T, C) float; we reshape to (B, T, H, N=head_size) for
    the kernel. Output is (B, T, C) in the caller's dtype (fla runs in
    bf16 internally; we cast back to avoid downstream GroupNorm dtype
    mismatches).
    """
    B, T, C = r.size()
    N = head_size
    H = C // N
    kernel_dtype = torch.bfloat16
    out_dtype = r.dtype

    # Convention translation: BlinkDL's CUDA kernel and slow reference apply
    # decay = exp(-exp(w_raw)) where w_raw = -softplus(-X) - 0.5 lives in
    # (-inf, -0.5]. fla's chunk_rwkv7 computes decay = exp(w_fla) internally
    # (the docstring calls w "log decay"). So we must pre-transform:
    #     w_fla = -exp(w_raw)  =>  exp(w_fla) = exp(-exp(w_raw))  ✓
    # Our own rwkv7_tmix.py trains under fla's convention natively, so
    # Aegir checkpoints don't need this translation — only loaded-from-
    # BlinkDL checkpoints do.
    w_translated = -torch.exp(w)

    r_ = rearrange(r.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    k_ = rearrange(k.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    v_ = rearrange(v.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    w_ = rearrange(w_translated.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    a_ = rearrange(a.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    b_ = rearrange(b.to(kernel_dtype), "b t (h d) -> b t h d", d=N).contiguous()
    with torch.amp.autocast("cuda", enabled=False):
        out, _ = chunk_rwkv7(r_, w_, k_, v_, a_, b_, output_final_state=False)
    return rearrange(out, "b t h d -> b t (h d)").to(out_dtype)


# ── Model (BlinkDL RWKV-7 verbatim, config auto-detected) ──────


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, C, head_size, d_decay, d_aaa, d_mv, d_gate, layer_id):
        super().__init__()
        self.head_size = head_size
        self.n_head = C // head_size
        self.layer_id = layer_id
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, d_decay))
        self.w2 = nn.Parameter(torch.empty(d_decay, C))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, d_aaa))
        self.a2 = nn.Parameter(torch.empty(d_aaa, C))
        # Layer 0 sets v_first, doesn't read it — skip v0/v1/v2.
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, C))
            self.v1 = nn.Parameter(torch.empty(C, d_mv))
            self.v2 = nn.Parameter(torch.empty(d_mv, C))
        self.g1 = nn.Parameter(torch.empty(C, d_gate))
        self.g2 = nn.Parameter(torch.empty(d_gate, C))
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(self.n_head, head_size))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, C, eps=64e-5)

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk * a, self.head_size)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
            dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, C, d_ffn):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, d_ffn, bias=False)
        self.value = nn.Linear(d_ffn, C, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class Block(nn.Module):
    def __init__(self, C, head_size, d_decay, d_aaa, d_mv, d_gate, d_ffn, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)
        self.att = RWKV_Tmix_x070(C, head_size, d_decay, d_aaa, d_mv, d_gate, layer_id)
        self.ffn = RWKV_CMix_x070(C, d_ffn)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_att, v_first = self.att(self.ln1(x), v_first)
        x = x + x_att
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV_x070(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        C = cfg["n_embd"]
        self.emb = nn.Embedding(cfg["vocab_size"], C)
        self.blocks = nn.ModuleList([
            Block(
                C, cfg["head_size"],
                cfg["d_decay"], cfg["d_aaa"], cfg["d_mv"], cfg["d_gate"],
                cfg["d_ffn"], i,
            ) for i in range(cfg["n_layer"])
        ])
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Linear(C, cfg["vocab_size"], bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for blk in self.blocks:
            x, v_first = blk(x, v_first)
        x = self.ln_out(x)
        return self.head(x)


# ── Config detection from state dict ───────────────────────────


def detect_config(state: dict) -> dict:
    n_layer = sum(1 for k in state if ".ln1.weight" in k and "blocks." in k)
    n_embd = state["emb.weight"].shape[1]
    vocab_size = state["emb.weight"].shape[0]
    d_decay = state["blocks.0.att.w1"].shape[1]
    d_aaa = state["blocks.0.att.a1"].shape[1]
    # v1/v2 only exist for layers ≥1 (layer 0 sets v_first, doesn't read it).
    d_mv = state["blocks.1.att.v1"].shape[1]
    d_gate = state["blocks.0.att.g1"].shape[1]
    d_ffn = state["blocks.0.ffn.key.weight"].shape[0]
    head_size = 64  # RWKV-7 standard; 0.1B through 13B all use 64
    n_head = n_embd // head_size
    assert state["blocks.0.att.r_k"].shape == (n_head, head_size), (
        f"r_k shape mismatch: expected ({n_head}, {head_size}), got "
        f"{state['blocks.0.att.r_k'].shape} — head_size may not be 64."
    )
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "head_size": head_size,
        "d_decay": d_decay,
        "d_aaa": d_aaa,
        "d_mv": d_mv,
        "d_gate": d_gate,
        "d_ffn": d_ffn,
    }


# ── Eval loop ──────────────────────────────────────────────────


def _iter_eval_bytes(data_dir: Path, max_bytes: int, skip_bytes: int) -> bytes:
    """Take the tail of the last shard (distinct from training byte range)."""
    shards = sorted(data_dir.glob("*.txt"))
    if not shards:
        raise FileNotFoundError(f"No .txt shards in {data_dir}")
    # Use the LAST shard's tail so training runs won't have seen these bytes
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
    ap.add_argument("--tokenizer", type=Path,
                    default=Path(
                        "/home/rch/local/src/oss/rwkv-lm/RWKV-v7/rwkv_vocab_v20230424.txt"
                    ))
    ap.add_argument("--data-dir", type=Path, default=Path("/raid/datasets/fineweb-edu"))
    ap.add_argument("--max-bytes", type=int, default=1_000_000,
                    help="Eval this many bytes from the tail of the last shard")
    ap.add_argument("--skip-bytes", type=int, default=0,
                    help="Skip this many bytes from the very tail before starting")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    log.info("Loading tokenizer: %s", args.tokenizer)
    tok = RWKV_TOKENIZER(str(args.tokenizer))

    log.info("Loading checkpoint: %s", args.checkpoint)
    t0 = time.time()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    log.info("  loaded %d tensors in %.1fs", len(state), time.time() - t0)
    cfg = detect_config(state)
    log.info("config: %s", cfg)

    log.info("Building model (%d params)...", cfg["n_layer"] * cfg["n_embd"] * cfg["n_embd"] * 10)
    model = RWKV_x070(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  actual params: %s (%.2fB)", f"{n_params:,}", n_params / 1e9)

    log.info("Loading state dict...")
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info("load stats: %d missing, %d unexpected", len(missing), len(unexpected))
    if missing:
        log.warning("  missing (first 10): %s", missing[:10])
    if unexpected:
        log.warning("  unexpected (first 10): %s", unexpected[:10])
    del state
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()

    log.info("Fetching eval bytes from FineWeb tail...")
    raw = _iter_eval_bytes(args.data_dir, args.max_bytes, args.skip_bytes)
    log.info("  got %d bytes (%.2f MB)", len(raw), len(raw) / 2**20)

    log.info("Tokenizing...")
    t0 = time.time()
    token_ids = tok.encodeBytes(raw)
    log.info("  %d tokens in %.1fs  (avg %.2f bytes/token)",
             len(token_ids), time.time() - t0, len(raw) / max(1, len(token_ids)))

    # Count total UTF-8 bytes covered by each token — needed for bits/byte.
    # (The tokenizer may skip/consume multi-byte codepoints; we want the
    # aggregate byte-coverage so the final metric is comparable to our
    # byte-level loss.)
    bytes_per_token: list[int] = [len(tok.idx2token[t]) for t in token_ids]
    total_bytes = sum(bytes_per_token)
    assert abs(total_bytes - len(raw)) < 256, (
        f"Byte accounting mismatch: tokens cover {total_bytes} vs raw {len(raw)}"
    )

    # Sliding-window eval. Each window: input=tokens[i:i+ctx], target=tokens[i+1:i+ctx+1].
    # Take loss on LAST HALF of each window only (after the first N/2, the
    # kv-state has stabilised, reducing underestimation from boundary effects).
    ctx = args.ctx_len
    stride = ctx // 2
    log.info("Running forward passes at ctx_len=%d, stride=%d...", ctx, stride)
    total_nll_nats = 0.0
    total_scored_tokens = 0
    total_scored_bytes = 0
    t0 = time.time()
    n_windows = 0

    with torch.no_grad():
        for start in range(0, max(1, len(token_ids) - ctx - 1), stride):
            input_ids = token_ids[start : start + ctx]
            target_ids = token_ids[start + 1 : start + ctx + 1]
            if len(target_ids) < ctx:
                break
            # Only score the last-half tokens for the non-first windows.
            score_start = 0 if start == 0 else ctx // 2
            in_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            tgt_t = torch.tensor([target_ids], dtype=torch.long, device=device)
            logits = model(in_t)  # (1, ctx, vocab)
            logp = F.log_softmax(logits[0].float(), dim=-1)
            nll_per_tok = -logp.gather(1, tgt_t[0, :, None]).squeeze(1)
            scored = nll_per_tok[score_start:]
            scored_tokens = target_ids[score_start:]
            scored_bytes = sum(bytes_per_token[start + 1 + score_start : start + 1 + ctx])
            total_nll_nats += scored.sum().item()
            total_scored_tokens += len(scored_tokens)
            total_scored_bytes += scored_bytes
            n_windows += 1
            if n_windows % 10 == 0:
                bpb = total_nll_nats / max(1, total_scored_bytes) / math.log(2)
                log.info("  window %3d  tokens=%d bytes=%d  running bits/byte=%.4f",
                         n_windows, total_scored_tokens, total_scored_bytes, bpb)

    elapsed = time.time() - t0
    bits_per_byte = total_nll_nats / max(1, total_scored_bytes) / math.log(2)
    nll_per_byte_nats = total_nll_nats / max(1, total_scored_bytes)
    nll_per_token_nats = total_nll_nats / max(1, total_scored_tokens)

    log.info("=" * 60)
    log.info("FINAL: %d windows, %d scored tokens, %d scored bytes, %.1fs",
             n_windows, total_scored_tokens, total_scored_bytes, elapsed)
    log.info("  nll/token (nats): %.4f", nll_per_token_nats)
    log.info("  nll/byte  (nats): %.4f", nll_per_byte_nats)
    log.info("  BITS / BYTE     : %.4f", bits_per_byte)
    log.info("=" * 60)

    result = {
        "checkpoint": str(args.checkpoint),
        "config": cfg,
        "params": n_params,
        "eval_bytes_total": len(raw),
        "scored_bytes": total_scored_bytes,
        "scored_tokens": total_scored_tokens,
        "nll_per_token_nats": nll_per_token_nats,
        "nll_per_byte_nats": nll_per_byte_nats,
        "bits_per_byte": bits_per_byte,
        "ctx_len": ctx,
        "stride": stride,
        "n_windows": n_windows,
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
