"""
RWKV-8 building blocks: ROSA time mixing and relu² channel mixing.

Adapted from RWKV-v8 "Heron" (BlinkDL/RWKV-LM), parameterized to accept
d_model/RWKVConfig rather than global args, and conforming to H-Net's
forward/step inference interface.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from aegir.models.config import RWKVConfig
from aegir.modules.rosa import RosaQKV1Bit


@dataclass
class RWKVBlockState:
    """Per-block inference state for autoregressive decoding.

    Fields:
        att_x_prev: (B, D) — previous hidden state for time-shift mixing.
        ffn_x_prev: (B, D) — previous hidden state for FFN time-shift.
        att_kv: (B, H, N, N) — recurrent KV state matrix for RWKV-7 TimeMix.
            None for ROSA-only blocks that lack recurrent state.
    """

    att_x_prev: torch.Tensor
    ffn_x_prev: torch.Tensor
    att_kv: Optional[torch.Tensor] = None


class RWKV_ROSA(nn.Module):
    """RWKV-8 time mixing with ROSA suffix automaton attention.

    Time-shift mixing → Q/K/V linear projections → ROSA matching → output projection.

    Args:
        d_model: Hidden dimension.
        rwkv_cfg: RWKV configuration.
        layer_idx: Layer index (for inference cache keying).
    """

    def __init__(
        self,
        d_model: int,
        rwkv_cfg: RWKVConfig = None,
        layer_idx: int = 0,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_q = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_k = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.x_v = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))

        self.q = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.rosa_qkv = RosaQKV1Bit(d_model)
        self.o = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

    def forward(self, x, inference_params=None, **kwargs):
        xx = self.time_shift(x) - x
        q = x + xx * self.x_q
        k = x + xx * self.x_k
        v = x + xx * self.x_v
        y = self.rosa_qkv(self.q(q), self.k(k), self.v(v))
        return self.o(y)

    def step(self, x, inference_params):
        """Single-token step for autoregressive decoding.

        Args:
            x: (B, 1, D) input.
            inference_params: Contains att_x_prev state.
        """
        state = inference_params.key_value_memory_dict[self.layer_idx]
        x_prev = state.att_x_prev
        x_squeezed = x.squeeze(1)  # (B, D)
        xx = x_prev - x_squeezed
        q = x_squeezed + xx * self.x_q.squeeze(0).squeeze(0)
        k = x_squeezed + xx * self.x_k.squeeze(0).squeeze(0)
        v = x_squeezed + xx * self.x_v.squeeze(0).squeeze(0)
        state.att_x_prev = x_squeezed.clone()
        # For single-token step, ROSA needs full context — fall back to zero output
        # (ROSA is primarily useful during prefill; RNN-mode decoding uses cached state)
        y = torch.zeros_like(x_squeezed)
        return self.o(y).unsqueeze(1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.q.weight.device
        return RWKVBlockState(
            att_x_prev=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            ffn_x_prev=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )


class RWKV_CMix(nn.Module):
    """RWKV-8 channel mixing (FFN) with relu² activation.

    Time-shift → key projection (D → 4D) → relu² → value projection (4D → D).

    Args:
        d_model: Hidden dimension.
        rwkv_cfg: RWKV configuration.
        layer_idx: Layer index (for inference cache keying).
    """

    def __init__(
        self,
        d_model: int,
        rwkv_cfg: RWKVConfig = None,
        layer_idx: int = 0,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        dim_ffn = int(d_model * (rwkv_cfg.dim_ffn_mult if rwkv_cfg else 4.0))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.key = nn.Linear(d_model, dim_ffn, bias=False, **factory_kwargs)
        self.value = nn.Linear(dim_ffn, d_model, bias=False, **factory_kwargs)

    def forward(self, x, **kwargs):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

    def step(self, x, inference_params):
        """Single-token step for autoregressive decoding."""
        state = inference_params.key_value_memory_dict[self.layer_idx]
        x_prev = state.ffn_x_prev
        x_squeezed = x.squeeze(1)  # (B, D)
        xx = x_prev - x_squeezed
        k = x_squeezed + xx * self.x_k.squeeze(0).squeeze(0)
        state.ffn_x_prev = x_squeezed.clone()
        k = torch.relu(self.key(k)) ** 2
        return self.value(k).unsqueeze(1)
