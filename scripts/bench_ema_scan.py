#!/usr/bin/env python3
"""Benchmark EMA scan variants for DeChunkLayer.

Compares:
  1. Current sequential _ema_scan (list + torch.stack)
  2. torch.compile of the same
  3. SSD-kernel via mamba_chunk_scan_combined
  4. Log-space parallel scan (cumsum)

Measures forward + backward at realistic shapes: (B=32, L=1024|2048, D=192).
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def ema_scan_sequential(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Current implementation from aegir.modules.dc._ema_scan."""
    outputs = [x[:, 0]]
    for t in range(1, x.shape[1]):
        outputs.append(decay[:, t] * outputs[-1] + (1 - decay[:, t]) * x[:, t])
    return torch.stack(outputs, dim=1)


def ema_scan_logspace(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Log-space parallel scan.

    y[t] = P[t] * (x[0] + sum_{s=1..t} (1-d[s]) * x[s] / P[s])
    where P[t] = prod_{u=1..t} d[u].

    fp32 internally; numerics degrade if decay→0 for many steps (log→-inf).
    Safe for decay in [1e-4, 1-1e-4] (the DeChunkLayer clamps).
    """
    B, L, D = x.shape
    d = decay.float().clamp(min=1e-6, max=1 - 1e-6)
    log_d = torch.log(d)
    # Cumulative log-product over timesteps 1..L (P[0] = 1, so cumsum starts at t=1)
    log_P = torch.cumsum(log_d[:, 1:], dim=1)  # (B, L-1, D)
    log_P = F.pad(log_P, (0, 0, 1, 0), value=0.0)  # (B, L, D)
    # v[0] = x[0], v[s>=1] = (1-d[s]) * x[s] / P[s] = (1-d[s]) * x[s] * exp(-log_P[s])
    x_f = x.float()
    v0 = x_f[:, :1]  # (B, 1, D)
    v_rest = (1 - d[:, 1:]) * x_f[:, 1:] * torch.exp(-log_P[:, 1:])
    v = torch.cat([v0, v_rest], dim=1)
    cum_v = torch.cumsum(v, dim=1)
    y = torch.exp(log_P) * cum_v
    return y.to(x.dtype)


def ema_scan_blelloch(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Parallel EMA scan via associative_scan (Blelloch).

    Represent each step s as a pair (d[s], v[s]) where:
      s = 0:  (d[0]=1, v[0]=x[0])              # identity multiplier → y[0] = x[0]
      s >= 1: (d[s], (1-d[s]) * x[s])

    Combine op:  (d1, v1) ⊕ (d2, v2) = (d1*d2, d2*v1 + v2)

    Then scanned pair at step t is (prod d[1..t], sum_{s=1..t} (prod d[s+1..t])*(1-d[s])*x[s] + prod d[1..t]*x[0]).
    Taking the second element gives y[t] directly.

    Numerically stable in bf16: coefficients stay in [0, 1], values stay bounded.
    """
    from torch._higher_order_ops.associative_scan import associative_scan

    if decay.dim() == 3 and decay.shape[-1] == 1:
        d = decay.expand_as(x).contiguous()
    else:
        d = decay
    # d_seq[:, 0] = 1 (identity), d_seq[:, t>0] = d[t]
    ones_head = torch.ones_like(d[:, :1])
    d_seq = torch.cat([ones_head, d[:, 1:]], dim=1)
    # v_seq[:, 0] = x[0], v_seq[:, t>0] = (1-d[t]) * x[t]
    v_seq = torch.cat([x[:, :1], (1 - d[:, 1:]) * x[:, 1:]], dim=1)

    def combine(a, b):
        d1, v1 = a
        d2, v2 = b
        return (d1 * d2, d2 * v1 + v2)

    # associative_scan expects dim argument and pytrees
    d_out, v_out = associative_scan(combine, (d_seq, v_seq), dim=1)
    return v_out


def ema_scan_ssd(x: torch.Tensor, decay: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """SSD kernel via mamba_chunk_scan_combined.

    Mapping (discretization matches ssd_chunk_scan_combined_ref):
      h[t] = exp(dt*A) * h[t-1] + dt * B * x[t]
      y[t] = C * h[t]

    To get y[t] = decay[t]*y[t-1] + (1-decay[t])*x[t]:
      A = -1 (scalar nheads-shaped)
      dt[t] = -log(decay[t])              → exp(dt*A) = decay[t]
      B[t] = (1 - decay[t]) / dt[t]        → dt*B = 1 - decay[t]
      C = 1

    The ratio (1-d)/(-log(d)) → 1 as d→1 (L'Hopital); clamp decay away from 1.
    """
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    B_sz, L, D = x.shape
    orig_dtype = x.dtype
    d = decay.float().clamp(min=1e-4, max=1 - 1e-4)
    if d.dim() == 3 and d.shape[-1] == 1:
        d = d.squeeze(-1)  # (B, L)
    elif d.dim() == 3 and d.shape[-1] == D:
        raise NotImplementedError("SSD path requires scalar decay per timestep")

    # Match sequential convention y[0] = x[0]: force dt[0]=0 → exp(0)=1 carryover
    # from initial_states=x[0]. For t>=1 use standard mapping.
    dt = -torch.log(d)                                  # (B, L)
    dt = torch.cat([torch.zeros_like(dt[:, :1]), dt[:, 1:]], dim=1)
    B_coef = torch.where(
        dt > 0, (1 - torch.exp(-dt)) / dt.clamp(min=1e-8), torch.zeros_like(dt)
    )

    x_ssm = x.unsqueeze(2).to(torch.bfloat16)           # (B, L, 1, D)
    dt_ssm = dt.unsqueeze(-1).to(torch.bfloat16)        # (B, L, 1)
    A_ssm = torch.tensor([-1.0], device=x.device)
    B_ssm = B_coef.to(torch.bfloat16).view(B_sz, L, 1, 1)
    C_ssm = torch.ones(B_sz, L, 1, 1, device=x.device, dtype=torch.bfloat16)

    # initial_states shape: (batch, nheads, headdim, dstate) = (B, 1, D, 1)
    init = x[:, 0].to(torch.bfloat16).view(B_sz, 1, D, 1)

    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        x_ssm = F.pad(x_ssm, (0, 0, 0, 0, 0, pad))
        dt_ssm = F.pad(dt_ssm, (0, 0, 0, pad))
        B_ssm = F.pad(B_ssm, (0, 0, 0, 0, 0, pad))
        C_ssm = F.pad(C_ssm, (0, 0, 0, 0, 0, pad))

    y = mamba_chunk_scan_combined(
        x_ssm, dt_ssm, A_ssm, B_ssm, C_ssm,
        chunk_size=chunk_size, initial_states=init,
    )
    y = y[:, :L, 0, :]
    return y.to(orig_dtype)


@contextmanager
def timed(label: str, n: int = 1, sync: bool = True):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:40s} {dt:8.3f} ms")


def bench(B: int, L: int, D: int, dtype: torch.dtype, device: str, *, warmup: int = 3, iters: int = 20):
    print(f"\n=== B={B}, L={L}, D={D}, dtype={dtype} ===")
    torch.manual_seed(0)
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)
    # Realistic decay: boundary probs from _ema_scan caller land in [1e-4, 1-1e-4]
    decay = torch.rand(B, L, 1, device=device, dtype=dtype).clamp(1e-3, 1 - 1e-3).expand(-1, -1, D).contiguous()

    # Reference: sequential (this is what we replace)
    y_ref = ema_scan_sequential(x, decay)
    print(f"  reference dtype: {y_ref.dtype}, shape: {y_ref.shape}")

    # Numerical equivalence checks
    for name, fn in [("blelloch", ema_scan_blelloch), ("logspace", ema_scan_logspace), ("ssd", ema_scan_ssd)]:
        try:
            # SSD only accepts scalar-per-timestep decay; all D channels equal in our test → collapse
            if name == "ssd":
                d_in = decay[:, :, :1]  # (B, L, 1) — will be squeezed inside
                # But SSD applies same A to all D channels → all D outputs identical per step.
                # Our decay has same value across D, so this matches. Good for test.
            else:
                d_in = decay
            y = fn(x, d_in)
            err = (y.float() - y_ref.float()).abs().max().item()
            rel = err / (y_ref.float().abs().max().item() + 1e-9)
            print(f"  {name:12s} max_abs_err={err:.3e}  max_rel_err={rel:.3e}")
        except Exception as e:
            print(f"  {name:12s} FAILED: {e}")

    # Forward timing
    print("  -- forward --")
    # sequential baseline
    for _ in range(warmup):
        _ = ema_scan_sequential(x, decay)
    with timed("sequential (current)", n=iters):
        for _ in range(iters):
            _ = ema_scan_sequential(x, decay)

    # compiled
    try:
        compiled = torch.compile(ema_scan_sequential, dynamic=False, fullgraph=False)
        for _ in range(warmup):
            _ = compiled(x, decay)
        with timed("torch.compile(sequential)", n=iters):
            for _ in range(iters):
                _ = compiled(x, decay)
    except Exception as e:
        print(f"  compile failed: {e}")

    try:
        for _ in range(warmup):
            _ = ema_scan_blelloch(x, decay)
        with timed("blelloch (associative_scan)", n=iters):
            for _ in range(iters):
                _ = ema_scan_blelloch(x, decay)
    except Exception as e:
        print(f"  blelloch failed: {e}")

    for _ in range(warmup):
        _ = ema_scan_logspace(x, decay)
    with timed("logspace (cumsum)", n=iters):
        for _ in range(iters):
            _ = ema_scan_logspace(x, decay)

    try:
        d_in_ssd = decay[:, :, :1]
        for _ in range(warmup):
            _ = ema_scan_ssd(x, d_in_ssd)
        with timed("SSD (mamba_chunk_scan)", n=iters):
            for _ in range(iters):
                _ = ema_scan_ssd(x, d_in_ssd)
    except Exception as e:
        print(f"  SSD failed: {e}")

    # Forward + backward
    print("  -- forward + backward --")
    def fb(fn, d):
        x2 = x.detach().clone().requires_grad_(True)
        y = fn(x2, d)
        y.sum().backward()
        return x2.grad

    for _ in range(warmup):
        _ = fb(ema_scan_sequential, decay)
    with timed("sequential fwd+bwd", n=iters):
        for _ in range(iters):
            _ = fb(ema_scan_sequential, decay)

    try:
        for _ in range(warmup):
            _ = fb(ema_scan_blelloch, decay)
        with timed("blelloch fwd+bwd", n=iters):
            for _ in range(iters):
                _ = fb(ema_scan_blelloch, decay)
    except Exception as e:
        print(f"  blelloch fwd+bwd failed: {e}")

    for _ in range(warmup):
        _ = fb(ema_scan_logspace, decay)
    with timed("logspace fwd+bwd", n=iters):
        for _ in range(iters):
            _ = fb(ema_scan_logspace, decay)

    try:
        for _ in range(warmup):
            _ = fb(ema_scan_ssd, decay[:, :, :1])
        with timed("SSD fwd+bwd", n=iters):
            for _ in range(iters):
                _ = fb(ema_scan_ssd, decay[:, :, :1])
    except Exception as e:
        print(f"  SSD fwd+bwd failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    assert torch.cuda.is_available(), "needs CUDA for SSD kernel"

    for B, L, D in [(32, 512, 192), (32, 1024, 192), (16, 2048, 192)]:
        bench(B, L, D, dtype, args.device)


if __name__ == "__main__":
    main()
