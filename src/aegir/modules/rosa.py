"""
ROSA (RWKV Online Suffix Automaton) — suffix-automaton-based exact sequence matching.

Ported from RWKV-v8 (BlinkDL/RWKV-LM). ROSA provides lossless infinite-range
retrieval by constructing an online suffix automaton over discretized (binarized)
hidden representations. The automaton runs on CPU; training uses counterfactual
perturbation through the binarization step.

Reference: ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching
           (arXiv:2602.02499)
"""

import torch
import torch.nn as nn


def rosa_qkv_ref(
    qqq: list[int], kkk: list[int], vvv: list[int]
) -> list[int]:
    """Pure-Python suffix automaton QKV matching (CPU).

    For each position i in the query sequence, finds the longest suffix of
    qqq[:i+1] that appears as a substring in kkk[:i], then returns the
    corresponding value from vvv at the position after the match.

    Args:
        qqq: Query token sequence.
        kkk: Key token sequence (builds the automaton).
        vvv: Value token sequence (provides retrieval targets).

    Returns:
        List of matched value tokens (-1 where no match found).
    """
    n = len(qqq)
    y = [-1] * n
    s = 2 * n + 1
    t = [None] * s
    f = [-1] * s
    m = [0] * s
    r = [-1] * s
    t[0] = {}
    g = 0
    u = 1
    w = h = 0
    assert n == len(kkk) == len(vvv)

    for i, (q, k) in enumerate(zip(qqq, kkk)):
        # Query phase: find longest matching suffix
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x
            p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        while v != -1 and (m[v] <= 0 or r[v] < 0):
            v = f[v]
        y[i] = vvv[r[v] + 1] if v != -1 else -1
        w, h = p, x

        # Key phase: extend suffix automaton with new character
        j = u
        u += 1
        t[j] = {}
        m[j] = m[g] + 1
        p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j
            p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u
                u += 1
                t[b] = t[d].copy()
                m[b] = m[p] + 1
                f[b] = f[d]
                r[b] = r[d]
                f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b
                    p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i
            v = f[v]

    return [max(0, yi) for yi in y]


def rosa_qkv_batch_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched ROSA QKV matching over the first dimension.

    Args:
        q: Query tensor, shape (batch, seq_len), dtype uint8.
        k: Key tensor, shape (batch, seq_len), dtype uint8.
        v: Value tensor, shape (batch, seq_len), dtype uint8.

    Returns:
        Matched indices tensor, shape (batch, seq_len), dtype uint8.
    """
    assert q.dtype == k.dtype == v.dtype == torch.uint8
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape == k.shape == v.shape
    qc = q.detach().contiguous().cpu()
    kc = k.detach().contiguous().cpu()
    vc = v.detach().contiguous().cpu()
    return torch.stack(
        [
            torch.as_tensor(
                rosa_qkv_ref(qq.tolist(), kk.tolist(), vv.tolist()),
                dtype=q.dtype,
            )
            for qq, kk, vv in zip(qc, kc, vc)
        ]
    ).to(q.device)


class _RosaQKV1BitOp(torch.autograd.Function):
    """Autograd function for 1-bit ROSA QKV matching.

    Binarizes Q/K/V activations per-channel, runs ROSA suffix matching,
    and reconstructs output using a learnable embedding scale.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c = q.shape
        # Binarize and reshape: (B, T, C) -> (B*C, T)
        qb = (q > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        kb = (k > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        vb = (v > 0).to(torch.uint8).transpose(1, 2).reshape(-1, t).contiguous()
        # Run ROSA per-channel, reshape back: (B*C, T) -> (B, T, C)
        idx = rosa_qkv_batch_ref(qb, kb, vb).view(b, c, t).transpose(1, 2).contiguous()
        # Reconstruct: matched 1 -> +emb, matched 0 -> -emb
        out = (2.0 * idx.to(q.dtype) - 1.0) * emb
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # ROSA is non-differentiable (suffix automaton).
        # Pass zero gradients for q, k, v; pass through for emb.
        return None, None, None, grad_output


class RosaQKV1Bit(nn.Module):
    """1-bit ROSA QKV layer.

    Binarizes activations per channel, performs suffix automaton matching,
    and scales output by a learnable embedding parameter.

    Args:
        d_model: Hidden dimension (number of channels).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Parameter(torch.full((1, 1, d_model), 1.0))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return _RosaQKV1BitOp.apply(q, k, v, self.emb)
