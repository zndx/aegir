"""
Unified block creation for Aegir's mixed architecture.

Block type codes:
  w/W = RWKV-7 full TimeMix (fla kernels) without/with SwiGLU MLP
  r   = RWKV-8 ROSA (suffix automaton) + CMix (relu²)
  R   = RWKV-8 ROSA (suffix automaton) + SwiGLU MLP
  t/T = Multi-head attention without/with SwiGLU MLP
  m/M = Mamba2 (SSM) without/with SwiGLU MLP (optional, requires mamba-ssm)
"""

from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor

from aegir.modules.mlp import SwiGLU
from aegir.modules.rwkv import RWKV_ROSA, RWKV_CMix


class LayerNormPrenorm(nn.LayerNorm):
    """LayerNorm with flash-attn RMSNorm-compatible prenorm interface.

    When called with prenorm=True, handles residual accumulation:
        residual = hidden_states + residual  (if residual provided)
        return norm(residual), residual
    """

    def forward(self, hidden_states, residual=None, prenorm=False, residual_in_fp32=False):
        if not prenorm:
            return super().forward(hidden_states)

        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states.float() if residual_in_fp32 else hidden_states
        return super().forward(hidden_states.to(self.weight.dtype)), residual


class Block(nn.Module):
    """Pre-norm block: RMSNorm → mixer → [RMSNorm → MLP].

    Handles residual connections via the prenorm pattern used by
    flash-attn's RMSNorm (returns hidden_states and residual separately).

    Args:
        d_model: Hidden dimension.
        mixer_cls: Callable returning the mixer module (Mamba2, MHA, or RWKV_ROSA).
        mlp_cls: Callable returning the MLP module (SwiGLU, RWKV_CMix, or Identity).
        norm_cls: Callable returning normalization module.
        residual_in_fp32: Whether to keep residual stream in fp32.
    """

    def __init__(
        self,
        d_model: int,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=None,
        residual_in_fp32: bool = True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        mixer_kwargs=None,
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, **mixer_kwargs
        )

        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def step(self, hidden_states, inference_params, residual=None):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = self.mixer.step(hidden_states, inference_params)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


def create_block(
    arch: str,
    d_model: int,
    d_intermediate: int = None,
    ssm_cfg: dict = None,
    attn_cfg: dict = None,
    rwkv_cfg=None,
    norm_epsilon: float = 1e-5,
    layer_idx: int = None,
    num_hidden_layers: int = None,
    residual_in_fp32: bool = True,
    device=None,
    dtype=None,
):
    """Create a Block with the specified architecture type.

    Args:
        arch: One of 'w', 'W', 'r', 'R', 't', 'T', 'm', 'M'.
        d_model: Hidden dimension.
        d_intermediate: MLP intermediate dimension (for SwiGLU).
        ssm_cfg: Mamba2 config dict.
        attn_cfg: Attention config dict.
        rwkv_cfg: RWKVConfig instance.
        norm_epsilon: Epsilon for RMSNorm.
        layer_idx: Layer index for cache keying.
        num_hidden_layers: Total layer count (for RWKV-7 init).
        residual_in_fp32: Keep residual in fp32.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = ssm_cfg or {}
    attn_cfg = attn_cfg or {}

    # Mixer selection
    if arch in ("w", "W"):
        from aegir.modules.rwkv7_tmix import RWKV7TimeMix

        mixer_cls = partial(
            RWKV7TimeMix,
            rwkv_cfg=rwkv_cfg,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            **factory_kwargs,
        )
    elif arch in ("t", "T"):
        from aegir.modules.mha import CausalMHA

        mixer_cls = partial(
            CausalMHA, **attn_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    elif arch in ("m", "M"):
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except ImportError:
            raise ImportError(
                "mamba-ssm is required for 'm'/'M' block types. "
                "Install it or use 'w'/'W' (RWKV-7) blocks instead."
            )

        class Mamba2Wrapper(Mamba2):
            def step(self, hidden_states, inference_params):
                conv_state, ssm_state = inference_params.key_value_memory_dict[
                    self.layer_idx
                ]
                result, conv_state, ssm_state = super().step(
                    hidden_states, conv_state, ssm_state
                )
                inference_params.key_value_memory_dict[self.layer_idx][0].copy_(
                    conv_state
                )
                inference_params.key_value_memory_dict[self.layer_idx][1].copy_(
                    ssm_state
                )
                return result

        mixer_cls = partial(
            Mamba2Wrapper, **ssm_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    elif arch in ("r", "R"):
        mixer_cls = partial(
            RWKV_ROSA,
            rwkv_cfg=rwkv_cfg,
            layer_idx=layer_idx,
            **factory_kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch!r}")

    # MLP selection
    if arch in ("r", "w"):
        # Lowercase RWKV: CMix (relu²) as FFN
        mlp_cls = partial(
            RWKV_CMix,
            rwkv_cfg=rwkv_cfg,
            layer_idx=layer_idx,
            **factory_kwargs,
        )
    elif arch in ("T", "M", "R", "W"):
        # Uppercase: SwiGLU MLP
        mlp_cls = partial(SwiGLU, d_intermediate=d_intermediate, **factory_kwargs)
    elif arch in ("t", "m"):
        # Lowercase non-RWKV: no MLP
        mlp_cls = nn.Identity
    else:
        raise ValueError(f"Unknown architecture type: {arch!r}")

    # Normalization — use flash_attn RMSNorm if available, else fallback
    try:
        from flash_attn.ops.triton.layer_norm import RMSNorm

        norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    except ImportError:
        norm_cls = partial(LayerNormPrenorm, eps=norm_epsilon, **factory_kwargs)

    return Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
    )
