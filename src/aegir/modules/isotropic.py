"""
Isotropic (non-hierarchical) sequence processor.

Processes sequences using a flat stack of mixed blocks (Mamba2, MHA, RWKV-8)
specified by an architecture layout string like "m4T1r4".

Adapted from H-Net (goombalab/hnet), extended with 'r'/'R' RWKV block types.
"""

import re
import copy
from dataclasses import dataclass, field
from typing import Optional

import optree

import torch
import torch.nn as nn

from aegir.modules.block import create_block
from aegir.modules.utils import get_seq_idx, get_stage_cfg
from aegir.models.config import AegirConfig


@dataclass
class IsotropicInferenceParams:
    """Inference state for autoregressive decoding through an Isotropic module."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )


class Isotropic(nn.Module):
    """Non-hierarchical sequence processor with mixed block types.

    Parses an architecture layout string (e.g. "m4T1r4") and creates
    the corresponding stack of blocks.

    Block type codes:
        m = Mamba2 (no MLP)
        M = Mamba2 + SwiGLU MLP
        t = Multi-head attention (no MLP)
        T = Multi-head attention + SwiGLU MLP
        r = RWKV-8 ROSA + CMix (relu²)
        R = RWKV-8 ROSA + SwiGLU MLP

    Args:
        config: AegirConfig instance.
        pos_idx: Position index within the current stage (0=encoder, 2=decoder).
        stage_idx: Hierarchy depth index.
    """

    def __init__(
        self,
        config: AegirConfig,
        pos_idx: int,
        stage_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        # Navigate to this stage's layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]

        # Parse layout string: e.g. "w4T1r4" -> [("w","4"), ("T","1"), ("r","4")]
        layout_parse = re.findall(r"([mMtTrRwW])(\d+)", arch_layout)

        # Count total layers for RWKV-7 position-based init
        total_layers = sum(int(n) for _, n in layout_parse)

        layers = []
        layer_idx = 0
        self.arch_full = []
        self.height = 0

        for arch, n_layer in layout_parse:
            n = int(n_layer)
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    rwkv_cfg=config.rwkv_cfg,
                    layer_idx=(layer_idx + i),
                    num_hidden_layers=total_layers,
                    **factory_kwargs,
                )
                for i in range(n)
            ]
            # Height = number of residual additions (blocks with MLP count as 2)
            if arch.islower():
                self.height += n
            else:
                self.height += 2 * n
            self.arch_full.extend([arch] * n)
            layer_idx += n

        self.layers = nn.ModuleList(layers)

        # Final normalization
        try:
            from flash_attn.ops.triton.layer_norm import RMSNorm

            self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)
        except ImportError:
            from aegir.modules.block import LayerNormPrenorm

            self.rmsnorm = LayerNormPrenorm(self.d_model, eps=1e-5, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        key_value_memory_dict = {}
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)

        if mask is not None:
            packed = False
            assert hidden_states.dim() == 3, "Hidden states must be (B, L, D) in unpacked mode"
        else:
            attn_mixer_kwargs.update(
                {"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen}
            )
            ssm_mixer_kwargs.update(
                {"seq_idx": get_seq_idx(cu_seqlens, device=hidden_states.device)}
            )
            packed = True

        # v_first container for RWKV-7 value-first sharing across layers
        v_first = [None]

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            elif arch in ("r", "R"):
                # ROSA blocks work in 3D (B, L, D) mode
                layer_mixer_kwargs = {}
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("w", "W"):
                # RWKV-7 TimeMix blocks work in 3D (B, L, D) mode
                layer_mixer_kwargs = {"v_first": v_first}
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            else:
                raise NotImplementedError(f"Unknown block type: {arch!r}")

            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Final norm (prenorm=False ignores residual)
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params):
        """Token-by-token step. hidden_states is (B, 1, D)."""
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(
                hidden_states, inference_params, residual=residual
            )
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )
        inference_params.seqlen_offset += 1
        return hidden_states
