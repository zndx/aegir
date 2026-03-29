"""
Aegir: Recursive hierarchical backbone with dynamic chunking.

Combines RWKV-8 (with ROSA) at the innermost stage with Mamba-2
encoder/decoder at outer stages, connected by content-dependent
dynamic chunking.

Adapted from H-Net (goombalab/hnet), with RWKV-8 block type support.
"""

from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn

from aegir.modules.isotropic import Isotropic, IsotropicInferenceParams
from aegir.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from aegir.modules.utils import apply_optimization_params
from aegir.models.config import AegirConfig


class STE(torch.autograd.Function):
    """Straight-Through Estimator for gradient flow through routing decisions."""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


@dataclass
class AegirState:
    """Recursive inference state for the Aegir backbone."""

    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["AegirState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


class Aegir(nn.Module):
    """Recursive hierarchical sequence model.

    At each non-innermost stage:
        encoder (Isotropic) → routing → chunk → main_network (Aegir) → dechunk → decoder (Isotropic)

    At the innermost stage:
        main_network only (Isotropic — typically with RWKV-8 blocks)

    The hierarchy is defined by ``config.arch_layout``, a recursive list like
    ``["m4", ["m4", ["r12"], "m4"], "m4"]``.

    Args:
        config: AegirConfig instance.
        stage_idx: Current depth in the hierarchy (0 = outermost).
    """

    def __init__(
        self,
        config: AegirConfig,
        stage_idx: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        # Navigate to this stage's layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError(
                f"arch_layout must have 1 (innermost) or 3 (encoder/main/decoder) elements, got {len(arch_layout)}"
            )

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = Aegir
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                **_pos_idx_dict,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)

            # Residual projection in fp32 for numerical stability
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        # Dimension padding for cross-stage dimension mismatch
        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name or name.endswith(".o"):
                        nn.init.normal_(
                            m.weight, mean=0.0, std=initializer_range / (n_residuals**0.5)
                        )
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        else:
            n_residuals += self.encoder.height + self.decoder.height
            for sub in (self.encoder, self.decoder):
                for name, m in sub.named_modules():
                    if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                        if "out_proj" in name or "fc2" in name or name.endswith(".o"):
                            nn.init.normal_(
                                m.weight,
                                mean=0.0,
                                std=initializer_range / (n_residuals**0.5),
                            )
                        else:
                            nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            self.main_network._init_weights(initializer_range, n_residuals)

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        if self.is_innermost:
            return AegirState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            return AegirState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
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
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        if inference_params is None:
            inference_params = AegirState(main_network_state=None)
        else:
            assert mask is not None, "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Encoder
        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        # Residual for skip connection
        hidden_states_for_residual = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_for_residual)

        # Dynamic chunking: predict boundaries and downsample
        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        # Process chunked sequence at next hierarchy level
        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states,
            cu_seqlens=next_cu_seqlens,
            max_seqlen=next_max_seqlen,
            mask=next_mask,
            inference_params=inference_params.main_network_state,
            **mixer_kwargs,
        )

        # Reconstruct full-length sequence via EMA
        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu_seqlens,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        # Residual connection with STE
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        # Decoder
        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs,
        )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]

    def step(self, hidden_states, inference_params):
        """Token-by-token step for autoregressive decoding."""
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
                ),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_module_state
        )
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, bpred_output.boundary_mask
        )

        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, inference_params.main_network_state
            )
        else:
            prev_boundary_predictions = []

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        hidden_states = hidden_states[..., :D]

        return hidden_states, [bpred_output, *prev_boundary_predictions]
