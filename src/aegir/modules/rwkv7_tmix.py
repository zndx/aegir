"""
RWKV-7 time mixing with flash-linear-attention triton kernels.

Full RWKV-7 time mixing with recurrent state matrix, supporting chunk-based
parallel training and token-by-token autoregressive inference. Uses fla's
optimized Triton kernels (chunk_rwkv7, fused_mul_recurrent_rwkv7).

Reference: RWKV-v8 "Heron" (BlinkDL/RWKV-LM), fla RWKV7Attention
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.token_shift import token_shift
from fla.modules.l2norm import l2_norm
from fla.modules.layernorm import GroupNorm
from fla.ops.rwkv7.chunk import chunk_rwkv7
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7

from aegir.models.config import RWKVConfig
from aegir.modules.rwkv import RWKVBlockState


class RWKV7TimeMix(nn.Module):
    """Full RWKV-7 time mixing with recurrent state.

    Implements the complete RWKV-7 attention mechanism:
    - Time-shift mixing with 6 coefficients (r, w, k, v, a, g)
    - Decay and gating via low-rank adaptation (LoRA)
    - Value-first sharing across layers (layer 0 sets, others lerp)
    - L2-normalized key attention with bonus term
    - GroupNorm output with gated projection

    Uses chunk_rwkv7 for training (parallel over chunks) and manual
    recurrence for single-token inference.

    Args:
        d_model: Hidden dimension.
        rwkv_cfg: RWKV configuration (head_size, LoRA dims).
        layer_idx: Layer index (controls v_first sharing and init).
        num_hidden_layers: Total layer count (for position-based init).
    """

    def __init__(
        self,
        d_model: int,
        rwkv_cfg: RWKVConfig = None,
        layer_idx: int = 0,
        num_hidden_layers: int = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers or 1

        head_size = rwkv_cfg.head_size if rwkv_cfg else 64
        self.head_size = head_size
        assert d_model % head_size == 0, f"d_model={d_model} not divisible by head_size={head_size}"
        self.num_heads = d_model // head_size

        # Compute LoRA dimensions (following fla's auto-calculation)
        factor = head_size / 64
        sqrt_d = math.sqrt(d_model)

        decay_low_rank_dim = (
            rwkv_cfg.decay_low_rank_dim if rwkv_cfg and rwkv_cfg.decay_low_rank_dim
            else max(32, int(round(2.5 * sqrt_d * factor / 32) * 32))
        )
        gate_low_rank_dim = (
            rwkv_cfg.gate_low_rank_dim if rwkv_cfg and rwkv_cfg.gate_low_rank_dim
            else max(32, int(round(5 * sqrt_d / 32) * 32))
        )
        a_low_rank_dim = (
            rwkv_cfg.a_low_rank_dim if rwkv_cfg and rwkv_cfg.a_low_rank_dim
            else max(32, int(round(2.5 * sqrt_d * factor / 32) * 32))
        )
        v_low_rank_dim = (
            rwkv_cfg.v_low_rank_dim if rwkv_cfg and rwkv_cfg.v_low_rank_dim
            else max(32, int(round(1.7 * sqrt_d * factor / 32) * 32))
        )

        # Time-shift mixing coefficients
        self.x_r = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_w = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_k = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_v = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_a = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_g = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))

        # Main projections
        self.r_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Decay LoRA: w = -softplus(-(w0 + tanh(w1(xw)) @ w2)) - 0.5
        self.w0 = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
        self.w1 = nn.Linear(d_model, decay_low_rank_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(decay_low_rank_dim, d_model, bias=False, **factory_kwargs)

        # Attention gate LoRA: a = sigmoid(a0 + a2(a1(xa)))
        self.a0 = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
        self.a1 = nn.Linear(d_model, a_low_rank_dim, bias=False, **factory_kwargs)
        self.a2 = nn.Linear(a_low_rank_dim, d_model, bias=False, **factory_kwargs)

        # Value-first LoRA (layer > 0): v += (v_first - v) * sigmoid(v0 + v2(v1(xv)))
        self.has_v_first = layer_idx > 0
        if self.has_v_first:
            self.v0 = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
            self.v1 = nn.Linear(d_model, v_low_rank_dim, bias=False, **factory_kwargs)
            self.v2 = nn.Linear(v_low_rank_dim, d_model, bias=False, **factory_kwargs)

        # Output gate LoRA: g = g2(sigmoid(g1(xg)))
        self.g1 = nn.Linear(d_model, gate_low_rank_dim, bias=False, **factory_kwargs)
        self.g2 = nn.Linear(gate_low_rank_dim, d_model, bias=False, **factory_kwargs)

        # Key normalization and attention parameters
        self.k_k = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.k_a = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, head_size, **factory_kwargs))

        # GroupNorm (one group per head)
        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            hidden_size=d_model,
            elementwise_affine=True,
            eps=head_size * 1e-5,
            bias=True,
            **factory_kwargs,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following RWKV-7 conventions."""
        layer_id = self.layer_idx
        n_layer = self.num_hidden_layers
        d = self.d_model

        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

        ddd = torch.arange(d, dtype=torch.float32) / d

        with torch.no_grad():
            self.x_r.data.copy_((1.0 - ddd.pow(0.2 * ratio_1_to_almost0)).view(1, 1, -1))
            self.x_w.data.copy_((1.0 - ddd.pow(0.9 * ratio_1_to_almost0)).view(1, 1, -1))
            self.x_k.data.copy_((1.0 - ddd.pow(0.9 * ratio_1_to_almost0 + 0.4 * ratio_0_to_1)).view(1, 1, -1))
            self.x_v.data.copy_((1.0 - ddd.pow(0.4 * ratio_1_to_almost0 + 0.6 * ratio_0_to_1)).view(1, 1, -1))
            self.x_a.data.copy_((1.0 - ddd.pow(0.9 * ratio_1_to_almost0)).view(1, 1, -1))
            self.x_g.data.copy_((1.0 - ddd.pow(0.2 * ratio_1_to_almost0)).view(1, 1, -1))

            # Decay bias — position-dependent
            decay = torch.arange(d, dtype=torch.float32)
            decay = -7 + 5 * (decay / max(d - 1, 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0.data.copy_(decay)

            # Key normalization
            self.k_k.data.fill_(0.85)
            self.k_a.data.fill_(1.0)

            # Bonus term — small random init
            self.r_k.data.normal_(0, 0.1)

            # Output projection — zero init for residual bypass
            nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        inference_params=None,
        v_first: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with chunk-based RWKV-7 attention.

        Args:
            x: (B, T, D) hidden states.
            inference_params: Optional inference state.
            v_first: Mutable list ``[tensor_or_None]`` for value-first sharing.
                Layer 0 stores its value tensor; subsequent layers lerp with it.
        """
        B, T, D = x.shape
        H, N = self.num_heads, self.head_size

        # Token shift: delta[t] = x[t-1] - x[t], delta[0] = -x[0]
        delta = token_shift(x)

        # Time-shift mixing
        xr = x + delta * self.x_r
        xw = x + delta * self.x_w
        xk = x + delta * self.x_k
        xv = x + delta * self.x_v
        xa = x + delta * self.x_a
        xg = x + delta * self.x_g

        # Main projections
        r = self.r_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # Decay LoRA (log space for chunk kernel)
        w = self.w2(torch.tanh(self.w1(xw)))
        w = -F.softplus(-(self.w0 + w)) - 0.5

        # Attention gate LoRA
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))

        # Output gate LoRA
        g = self.g2(torch.sigmoid(self.g1(xg)))

        # Value-first sharing
        if v_first is not None:
            if self.layer_idx == 0:
                v_first[0] = v
            elif self.has_v_first:
                v = v + (v_first[0] - v) * torch.sigmoid(
                    self.v0 + self.v2(self.v1(xv))
                )

        # Key normalization: L2-norm of (k * k_k) per head
        kk = l2_norm(rearrange(k * self.k_k, "b t (h d) -> b t h d", d=N))

        # Key update with attention gate
        k = k * (1 + (a - 1) * self.k_a)

        # Reshape for kernel: (B, T, D) -> (B, T, H, N)
        # Cast all to bf16 for fla kernel compatibility under AMP
        kernel_dtype = torch.bfloat16 if r.is_cuda else r.dtype
        r = rearrange(r.to(kernel_dtype), "b t (h d) -> b t h d", d=N)
        k = rearrange(k.to(kernel_dtype), "b t (h d) -> b t h d", d=N)
        v = rearrange(v.to(kernel_dtype), "b t (h d) -> b t h d", d=N)
        w = rearrange(w.to(kernel_dtype), "b t (h d) -> b t h d", d=N)
        a_4d = rearrange(a.to(kernel_dtype), "b t (h d) -> b t h d", d=N)
        kk = kk.to(kernel_dtype)

        # Get initial recurrent state if available
        initial_state = None
        if inference_params is not None:
            state = inference_params.key_value_memory_dict[self.layer_idx]
            if state.att_kv is not None:
                initial_state = state.att_kv

        # Chunk RWKV-7 kernel (disable autocast to avoid dtype conflicts)
        with torch.amp.autocast("cuda", enabled=False):
            o, final_state = chunk_rwkv7(
                r, w, k, v,
                -kk, kk * a_4d,
                initial_state=initial_state,
                output_final_state=True,
            )

        # Store final recurrent state
        if inference_params is not None:
            state.att_kv = final_state

        # GroupNorm
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.g_norm(o)

        # Bonus term: sum(r * k * r_k, dim=-1, keepdim=True) * v
        bonus = (r * k * self.r_k).sum(dim=-1, keepdim=True) * v
        bonus = rearrange(bonus, "b t h d -> b t (h d)")
        o = o + bonus

        # Gate and output projection
        o = o * g
        return self.o_proj(o)

    def step(
        self,
        x: torch.Tensor,
        inference_params,
        v_first: Optional[list] = None,
    ) -> torch.Tensor:
        """Single-token autoregressive step.

        Args:
            x: (B, 1, D) input.
            inference_params: Inference state with att_x_prev and att_kv.
            v_first: Mutable list for value-first sharing.
        """
        B, _, D = x.shape
        H, N = self.num_heads, self.head_size

        state = inference_params.key_value_memory_dict[self.layer_idx]
        x_squeezed = x.squeeze(1)  # (B, D)
        x_prev = state.att_x_prev

        # Time shift (manual for single token)
        xx = x_prev - x_squeezed
        xr = x_squeezed + xx * self.x_r.squeeze(0).squeeze(0)
        xw = x_squeezed + xx * self.x_w.squeeze(0).squeeze(0)
        xk = x_squeezed + xx * self.x_k.squeeze(0).squeeze(0)
        xv = x_squeezed + xx * self.x_v.squeeze(0).squeeze(0)
        xa = x_squeezed + xx * self.x_a.squeeze(0).squeeze(0)
        xg = x_squeezed + xx * self.x_g.squeeze(0).squeeze(0)
        state.att_x_prev = x_squeezed.clone()

        # Projections
        r = self.r_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # Decay (actual multiplicative factor for step mode)
        w = self.w2(torch.tanh(self.w1(xw)))
        w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + w).float()))

        # Attention gate
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))

        # Output gate
        g = self.g2(torch.sigmoid(self.g1(xg)))

        # Value-first sharing
        if v_first is not None:
            if self.layer_idx == 0:
                v_first[0] = v
            elif self.has_v_first:
                v = v + (v_first[0] - v) * torch.sigmoid(
                    self.v0 + self.v2(self.v1(xv))
                )

        # Key normalization
        kk = F.normalize(
            (k * self.k_k).view(B, H, N), dim=-1, p=2.0
        ).view(B, H * N)
        k = k * (1 + (a - 1) * self.k_a)

        # Reshape to per-head
        r_h = r.view(B, H, N)
        k_h = k.view(B, H, N)
        v_h = v.view(B, H, N)
        w_h = w.view(B, H, N)
        kk_h = kk.view(B, H, N)
        a_h = a.view(B, H, N)

        # Manual recurrence: S[t] = S[t-1] * w + S[t-1] @ ab + vk
        vk = v_h.unsqueeze(-1) @ k_h.unsqueeze(-2)  # (B, H, N, N)
        ab = (-kk_h).unsqueeze(-1) @ (kk_h * a_h).unsqueeze(-2)  # (B, H, N, N)

        kv_state = state.att_kv  # (B, H, N, N)
        kv_state = (
            kv_state * w_h.unsqueeze(-2)
            + kv_state @ ab.float()
            + vk.float()
        )
        state.att_kv = kv_state

        # Read output: o = S @ r
        o = (kv_state.to(x.dtype) @ r_h.unsqueeze(-1)).squeeze(-1)  # (B, H, N)
        o = o.view(B, H * N)

        # GroupNorm
        o = self.g_norm(o)

        # Bonus term
        bonus = (r_h * k_h * self.r_k).sum(dim=-1, keepdim=True) * v_h
        o = o + bonus.view(B, H * N)

        # Gate and output projection
        o = o * g
        return self.o_proj(o).unsqueeze(1)

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        device = self.r_proj.weight.device
        return RWKVBlockState(
            att_x_prev=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
            att_kv=torch.zeros(
                batch_size, self.num_heads, self.head_size, self.head_size,
                device=device, dtype=torch.float32,
            ),
            ffn_x_prev=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )
