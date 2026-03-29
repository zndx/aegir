"""
Dynamic chunking: content-dependent hierarchical segmentation.

Adapted from H-Net (goombalab/hnet). Implements:
  - RoutingModule: predicts boundary probabilities via cosine similarity
  - ChunkLayer: downsamples by selecting boundary tokens
  - DeChunkLayer: reconstructs full sequence via EMA scan
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegir.modules.utils import get_seq_idx


def _ema_scan(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """EMA scan: y[t] = decay[t] * y[t-1] + (1 - decay[t]) * x[t].

    Args:
        x: (B, L, D) input values.
        decay: (B, L, 1) or (B, L, D) decay factors in [0, 1].

    Returns:
        (B, L, D) EMA output.
    """
    outputs = [x[:, 0]]
    for t in range(1, x.shape[1]):
        outputs.append(decay[:, t] * outputs[-1] + (1 - decay[:, t]) * x[:, t])
    return torch.stack(outputs, dim=1)


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """Inference state for the routing module.

    Attributes:
        has_seen_tokens: (batch_size,) bool — whether each sample has been processed.
        last_hidden_state: (batch_size, d_model) — last hidden state for boundary prediction.
    """

    has_seen_tokens: torch.Tensor
    last_hidden_state: torch.Tensor


@dataclass
class DeChunkState:
    """Inference state for the dechunk layer.

    Attributes:
        last_value: (batch_size, d_model) — last EMA value for reconstruction.
    """

    last_value: torch.Tensor


class RoutingModule(nn.Module):
    """Predicts chunk boundaries using cosine similarity between consecutive states.

    Higher dissimilarity = higher boundary probability. Q/K projections are
    initialized to identity to start with raw similarity.

    Args:
        d_model: Hidden dimension.
    """

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert mask is not None, "Mask must be provided if inference_params is provided"
            assert (
                ~inference_params.has_seen_tokens
            ).all(), "Cannot have seen tokens during prefill"

        if cu_seqlens is not None:
            hidden_states = hidden_states.unsqueeze(0)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1

        if mask is not None:
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(hidden_states.shape[0], device=hidden_states.device),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(self, hidden_states, inference_params):
        """Single-token boundary prediction. hidden_states is (B, 1, D)."""
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class ChunkLayer(nn.Module):
    """Downsamples a sequence by selecting only boundary tokens."""

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int(
                (next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max()
            )
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    """Reconstructs full-length sequence from chunk outputs via EMA.

    Boundary probability controls the blend: high p = use chunk output,
    low p = carry forward previous value. Uses a sequential EMA scan
    (torch.compile-friendly, no external CUDA kernels required).

    Args:
        d_model: Hidden dimension.
    """

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is not None:
            assert mask is not None, "Mask must be provided if inference_params is provided"
            assert boundary_mask[:, 0].all(), "First token must be a boundary during prefill"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
        else:
            B, L = boundary_mask.shape

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )

        original_dtype = hidden_states.dtype
        x = hidden_states.to(self.dtype)
        # decay = 1 - p: how much of the previous value to keep
        decay = (1 - p).unsqueeze(-1).to(self.dtype)

        out = _ema_scan(x, decay)

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        """Single-token dechunk step."""
        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
