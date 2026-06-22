#!/usr/bin/env python
"""Continue-pretrain a vanilla RWKV-7 (warm-started official ckpt, World tokenizer) on a World-v3 subsample
± our synthetic augmentation, using the FUSED ``rwkv7_clampw`` CUDA kernel (compiled via the nix CUDA
toolchain exposed as ``AEGIR_CUDA_*`` by devenv). **Path A**: isolate the DATA value of the augmentation
holding architecture fixed — control (α=0, base only) vs treatment (α>0, base⊕aug). Eval separately with
``eval_rwkv7_baseline`` (general) + ``eval_edge_probe`` (relational).

    uv run --no-sync python scripts/continue_pretrain_rwkv7.py \
        --load /raid/datasets/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
        --base /raid/build/aegir/path-a/base/subsample_100k \
        --aug  /raid/build/aegir/path-a/aug/pathA_aug \
        --alpha 0.02 --tokens 110000000 --ctx-len 2048 --batch 8 --lr 6e-5 \
        --out /raid/build/aegir/path-a/out/a02
    # control arm: --alpha 0  (base only; --aug ignored).

Reuses the model from ``eval_rwkv7_baseline`` but swaps its (fla) ``RWKV7_OP`` for the fused kernel — both
compute decay = exp(-exp(w_raw)), so the BlinkDL/raw-w convention is passed straight through (no fla
translation). bf16 throughout (the fused kernel requires it); modest LR + grad-clip for a warm-started
continue. Note for SCALE: add fp32 master weights / DeepSpeed; this is the validation/baseline rig.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── put the nix CUDA toolchain on LD_LIBRARY_PATH *before* importing torch (re-exec once if missing) ──
_cuda = os.environ.get("AEGIR_CUDA_HOME")
if _cuda:
    _need = f"{_cuda}/lib64"
    if _need not in os.environ.get("LD_LIBRARY_PATH", ""):
        # APPEND (never replace — dropping nix glibc breaks ninja + nix binaries), then re-exec.
        os.environ["LD_LIBRARY_PATH"] = f"{_need}:{REPO}/build/cuda-driver-libs:" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["CUDA_HOME"] = _cuda
        os.execv(sys.executable, [sys.executable, *sys.argv])
    os.environ.setdefault("CUDA_HOME", _cuda)
    os.environ["PATH"] = f"{_cuda}/bin:" + os.environ.get("PATH", "")
    _ccbin = os.environ.get("AEGIR_CUDA_CCBIN", "")
    if _ccbin:  # nvcc -ccbin follows CC; gcc-14 (nvcc-compatible), all on nix glibc
        os.environ["CC"] = f"{_ccbin}/gcc"
        os.environ["CXX"] = f"{_ccbin}/g++"
        os.environ["CUDAHOSTCXX"] = f"{_ccbin}/g++"
os.environ["RWKV_HEAD_SIZE"] = "64"
os.environ["RWKV_MY_TESTING"] = "x070"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import eval_rwkv7_baseline as E  # noqa: E402 — RWKV-7 model + detect_config + tokenizer (fla forward)

# ── fused rwkv7_clampw kernel (compiled via the nix toolchain); swaps fla's RWKV7_OP ──
_TRAIN = "/home/rch/local/src/oss/rwkv-lm/RWKV-v7/train_temp"
CHUNK_LEN = 16
from torch.utils.cpp_extension import load as _load  # noqa: E402
_load(name="rwkv7_clampw",
      sources=[f"{_TRAIN}/cuda/rwkv7_clampw.cu", f"{_TRAIN}/cuda/rwkv7_clampw.cpp"],
      is_python_module=False, verbose=False,
      extra_cuda_cflags=['-res-usage', '-D_N_=64', f'-D_CHUNK_LEN_={CHUNK_LEN}',
                         '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'])


class _ClampW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, N = r.shape
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
        torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = (torch.empty_like(x) for x in (r, w, k, v, a, b))
        torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy.bfloat16().contiguous(), s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db


def _fused_op(r, w, k, v, a, b, head_size):
    """Drop-in for eval_rwkv7_baseline.RWKV7_OP, but on the fused kernel (raw-w / BlinkDL convention)."""
    B, T, C = r.shape
    H = C // head_size
    z = lambda x: x.bfloat16().contiguous().view(B, T, H, head_size)  # noqa: E731
    return _ClampW.apply(z(r), z(w), z(k), z(v), z(a), z(b)).view(B, T, C)


E.RWKV7_OP = _fused_op  # the model's Tmix now drives the fused training kernel


def _hidden(model, idx):
    """RWKV_x070 forward up to ln_out (pre-head) — lets the head+CE be chunked."""
    x = model.emb(idx)
    v_first = torch.empty_like(x)
    for blk in model.blocks:
        x, v_first = blk(x, v_first)
    return model.ln_out(x)


def _chunked_ce_backward(model, hidden, y, chunk):
    """Memory-efficient next-token CE: apply head + CE in token-chunks with per-chunk backward into a
    DETACHED hidden, then propagate once through the backbone. Never materializes the full (B,T,vocab)
    logits — the ~3GB memory wall at 65536 vocab — so batch/model can grow. Math = plain mean CE: each
    chunk contributes (sum_CE / n) and its backward accumulates the head grad + the per-token hidden grad
    (hd.grad); `hidden.backward(hd.grad)` then carries that through ln_out/blocks/emb."""
    B, T, C = hidden.shape
    n = B * T
    hd = hidden.detach().requires_grad_(True)
    hf, yf = hd.reshape(n, C), y.reshape(n)
    total = 0.0
    for i in range(0, n, chunk):
        l = F.cross_entropy(model.head(hf[i:i + chunk]).float(), yf[i:i + chunk], reduction="sum") / n
        l.backward()
        total += float(l.detach())
    hidden.backward(hd.grad.reshape(B, T, C))
    return total


class MixedBinidx(torch.utils.data.Dataset):
    """Flat-stream window sampler over base⊕aug binidx; each item is drawn from aug with prob α else base."""

    def __init__(self, base: str, aug: str, alpha: float, ctx: int, n: int, seed: int = 42):
        self.base = np.memmap(base + ".bin", dtype=np.uint16, mode="r")
        self.aug = np.memmap(aug + ".bin", dtype=np.uint16, mode="r") if (aug and alpha > 0) else None
        self.alpha, self.ctx, self.n, self.seed = alpha, ctx, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        buf = self.aug if (self.aug is not None and rng.random() < self.alpha) else self.base
        off = rng.randint(0, len(buf) - self.ctx - 2)
        w = np.asarray(buf[off:off + self.ctx + 1], dtype=np.int64)
        return torch.from_numpy(w[:-1]), torch.from_numpy(w[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, help="warm-start checkpoint (.pth)")
    ap.add_argument("--base", required=True, help="base binidx prefix (World-v3 subsample)")
    ap.add_argument("--aug", default="", help="augmentation binidx prefix (our corpus)")
    ap.add_argument("--alpha", type=float, default=0.0, help="P(draw from aug); 0 = control")
    ap.add_argument("--tokens", type=int, default=110_000_000)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ce-chunk", type=int, default=2048, help="token-chunk for memory-efficient head+CE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    assert a.ctx_len % CHUNK_LEN == 0, f"ctx-len must be a multiple of {CHUNK_LEN}"
    torch.manual_seed(a.seed)
    dev = "cuda"

    state = torch.load(a.load, map_location="cpu", weights_only=True)
    cfg = E.detect_config(state)
    print(f"cfg: {cfg}", flush=True)
    model = E.RWKV_x070(cfg).to(device=dev, dtype=torch.bfloat16)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"warm-start {Path(a.load).name}: {len(miss)} missing / {len(unexp)} unexpected", flush=True)
    model.train()

    steps = a.tokens // (a.batch * a.ctx_len)
    ds = MixedBinidx(a.base, a.aug, a.alpha, a.ctx_len, steps * a.batch, a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        prog = (step - a.warmup) / max(1, steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"continue: α={a.alpha} tokens={a.tokens} steps={steps} batch={a.batch} ctx={a.ctx_len} lr={a.lr}", flush=True)
    metrics, t0, tok, step = [], time.time(), 0, 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        h = _hidden(model, x)
        loss_v = _chunked_ce_backward(model, h, y, a.ce_chunk)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        tok += x.numel()
        step += 1
        if step % a.log_every == 0:
            tps = tok / (time.time() - t0)
            print(f"step {step}/{steps} loss {loss_v:.4f} lr {lr_at(step):.2e} gnorm {gn:.2f} tok/s {tps:.0f}", flush=True)
            metrics.append({"step": step, "loss": loss_v, "lr": lr_at(step), "gnorm": float(gn), "tok_s": tps})
        if step % a.ckpt_every == 0:
            torch.save(model.state_dict(), f"{a.out}/ckpt_{step}.pth")
        if step >= steps:
            break
    torch.save(model.state_dict(), f"{a.out}/final.pth")
    Path(f"{a.out}/metrics.json").write_text(json.dumps(
        {"alpha": a.alpha, "tokens": a.tokens, "steps": steps, "ctx_len": a.ctx_len, "batch": a.batch, "lr": a.lr,
         "final_loss": (metrics[-1]["loss"] if metrics else None), "log": metrics}, indent=1))
    print(f"DONE α={a.alpha} steps={steps} final_loss={metrics[-1]['loss'] if metrics else '?'} → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())
