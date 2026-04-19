"""
Model heads: language modeling and column annotation task wrappers.

AegirForCausalLM — pretraining head (embeddings → backbone → lm_head)
AegirForColumnAnnotation — CTA/CPA finetuning head with role embeddings
"""

from collections import namedtuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from aegir.models.aegir import Aegir, AegirState
from aegir.models.config import AegirConfig
from aegir.modules.dc import RoutingModuleOutput
from aegir.modules.utils import apply_optimization_params


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: AegirState


@dataclass
class ColumnAnnotationOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]


@dataclass
class DEDOutput:
    """Cross-table data-element-discovery head output.

    Attributes:
        embeddings: (B, N, proj_dim) L2-normalized column embeddings.
            N is the padded number of column-CLS positions per sample;
            valid columns are flagged by ``col_mask``.
        col_mask: (B, N) bool — True for real columns, False for padding.
        bpred_output: per-stage RoutingModule outputs from the backbone.
    """

    embeddings: torch.Tensor
    col_mask: torch.Tensor
    bpred_output: list[RoutingModuleOutput]


class AegirForCausalLM(nn.Module):
    """Language modeling head for pretraining.

    embeddings → Aegir backbone → lm_head

    Args:
        config: AegirConfig instance.
    """

    def __init__(
        self,
        config: AegirConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        vocab_size = config.vocab_size
        d_embed = config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)
        self.backbone = Aegir(
            config=config,
            stage_idx=0,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def init_weights(self, initializer_range: float = 0.02) -> None:
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        self.backbone._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        for param in self.embeddings.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.lm_head.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.backbone._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """Forward pass for language modeling.

        Args:
            input_ids: (B, L) token IDs.
            mask: (B, L) boolean mask for unpacked mode.
            num_last_tokens: If > 0, only compute logits for last n tokens.
        """
        hidden_states = self.embeddings(input_ids)
        B, L, D = hidden_states.shape

        assert position_ids is None, (
            "Position ids are not supported due to hierarchical subsampling"
        )

        if mask is None:
            assert inference_params is None, "Inference params not supported in packed mode"
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutputTuple = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutputTuple(
            logits=lm_logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
        )

    def step(self, input_ids, inference_params):
        """Single-token step for autoregressive decoding."""
        B = input_ids.shape[0]
        assert B == 1, "Step currently only supports batch size 1"

        hidden_states = self.embeddings(input_ids)
        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )


def _extract_at_indexes(hidden_states: torch.Tensor, cls_indexes: torch.Tensor) -> torch.Tensor:
    """Extract hidden states at specified positions.

    Args:
        hidden_states: (B, L, D) tensor.
        cls_indexes: (B,) or (B, K) position indices.

    Returns:
        Extracted states, (B, D) or (B, K, D).
    """
    if cls_indexes.dim() == 1:
        return hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            cls_indexes,
        ]
    else:
        B, K = cls_indexes.shape
        batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(B, K)
        return hidden_states[batch_idx, cls_indexes]


class AegirForColumnAnnotation(nn.Module):
    """Column annotation head for CTA/CPA finetuning.

    Adds role embeddings (target vs context columns) and a classification head
    on top of the Aegir backbone. Follows the REVEAL paradigm of context-aware
    column encoding.

    Args:
        config: AegirConfig with num_labels > 0.
        max_roles: Maximum number of role types (target=0, context=1, 2, ...).
    """

    def __init__(
        self,
        config: AegirConfig,
        max_roles: int = 32,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_embed = config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.embeddings = nn.Embedding(config.vocab_size, d_embed, **factory_kwargs)
        self.role_embeddings = nn.Embedding(max_roles, d_embed, **factory_kwargs)
        self.backbone = Aegir(
            config=config,
            stage_idx=0,
            **factory_kwargs,
        )
        self.pooler = nn.Linear(d_embed, d_embed, **factory_kwargs)
        self.classifier = nn.Linear(d_embed, config.num_labels, **factory_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: torch.Tensor,
        cls_indexes: torch.Tensor,
        mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **mixer_kwargs,
    ) -> ColumnAnnotationOutput:
        """Forward pass for column annotation.

        Args:
            input_ids: (B, L) token IDs.
            role_ids: (B, L) role IDs (0=target, 1+=context columns).
            cls_indexes: (B,) position of target column's CLS token.
            mask: (B, L) boolean mask.
            labels: (B,) ground truth labels (optional, for convenience).
        """
        hidden_states = self.embeddings(input_ids) + self.role_embeddings(role_ids)
        B, L, D = hidden_states.shape

        if mask is None:
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        # Extract target column representation
        pooled = _extract_at_indexes(hidden_states, cls_indexes)
        pooled = torch.tanh(self.pooler(pooled))
        logits = self.classifier(pooled)

        return ColumnAnnotationOutput(logits=logits, bpred_output=bpred_output)


class AegirForDED(nn.Module):
    """Cross-table Data Element Discovery head.

    Pools per-column representations from packed multi-table input and
    projects them into an embedding space suitable for supervised
    contrastive training (Khosla et al., "Supervised Contrastive Learning",
    NeurIPS 2020). The head itself emits L2-normalized embeddings; the
    contrastive loss lives in aegir.utils.train so the training loop can
    plug into any clustering target (DBpedia type, synthetic ontology entity,
    hand-labeled data element).

    Column pooling uses the ``cls_indexes`` pattern already established for
    AegirForColumnAnnotation, but here cls_indexes is 2D (B, N_cols) so a
    single forward pass produces N embeddings per sample. Padding columns
    (encoded as ``cls_index = -1``) are zeroed in the output and flagged in
    ``col_mask`` so the loss can skip them cleanly.

    Args:
        config: AegirConfig. ``num_labels`` is ignored; use ``proj_dim``.
        proj_dim: contrastive embedding dimensionality. 128 is standard
            (MoCo, SimCLR, SupCon). Defaults to 128.
        max_roles: Maximum distinct role IDs (one per column/table).
    """

    def __init__(
        self,
        config: AegirConfig,
        proj_dim: int = 128,
        max_roles: int = 64,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.proj_dim = proj_dim
        d_embed = config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, d_embed, **factory_kwargs)
        self.role_embeddings = nn.Embedding(max_roles, d_embed, **factory_kwargs)
        self.backbone = Aegir(config=config, stage_idx=0, **factory_kwargs)
        self.pooler = nn.Linear(d_embed, d_embed, **factory_kwargs)
        # Two-layer MLP head — standard in contrastive pretraining literature.
        # Acts as a "view-specific" projection that gets dropped at inference
        # time (use pooler output for downstream clustering).
        self.proj = nn.Sequential(
            nn.Linear(d_embed, d_embed, **factory_kwargs),
            nn.GELU(),
            nn.Linear(d_embed, proj_dim, **factory_kwargs),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: torch.Tensor,
        cls_indexes: torch.Tensor,
        mask: torch.Tensor = None,
        **mixer_kwargs,
    ) -> DEDOutput:
        """Forward pass returning per-column L2-normalized embeddings.

        Args:
            input_ids: (B, L) token IDs.
            role_ids: (B, L) per-token role IDs. Role 0 = padding/global,
                roles 1..N_tables distinguish concatenated tables.
            cls_indexes: (B, N) column-CLS positions. Use ``-1`` for padding
                columns; those rows are zeroed and flagged in col_mask.
            mask: (B, L) bool. None switches to packed cu_seqlens mode.
        """
        hidden_states = self.embeddings(input_ids) + self.role_embeddings(role_ids)
        B, L, D = hidden_states.shape

        if mask is None:
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            **mixer_kwargs,
        )
        hidden_states = hidden_states.view(B, L, D)

        # Clamp -1 padding to 0 before gather, then mask out the resulting
        # junk rows with col_mask. Gather on a valid index is cheaper than
        # boolean scatter with ragged N per sample.
        col_mask = cls_indexes >= 0
        safe_idx = cls_indexes.clamp(min=0)
        pooled = _extract_at_indexes(hidden_states, safe_idx)  # (B, N, D)
        pooled = torch.tanh(self.pooler(pooled))
        proj = self.proj(pooled)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        proj = proj * col_mask.unsqueeze(-1)

        return DEDOutput(embeddings=proj, col_mask=col_mask, bpred_output=bpred_output)
