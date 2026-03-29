"""Shared utilities for Aegir modules."""

from dataclasses import asdict

import torch


def get_seq_idx(cu_seqlens: torch.Tensor, device=None) -> torch.Tensor:
    """Convert cumulative sequence lengths to per-token sequence indices.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape (num_seqs + 1,).
        device: Target device.

    Returns:
        Sequence index per token, shape (1, total_tokens), dtype int32.
    """
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()
    return seq_idx


def get_stage_cfg(cfg, stage_idx: int) -> dict:
    """Extract stage-specific config values from a per-stage config dataclass.

    For list-valued fields, selects the element at stage_idx.
    For scalar fields, passes through unchanged.
    """
    result = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, list) and len(v) > stage_idx:
            result[k] = v[stage_idx]
        elif isinstance(v, list):
            result[k] = None
        else:
            result[k] = v
    return result


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """Annotate a parameter with optimization metadata (lr_multiplier, weight_decay, etc.).

    Updates the parameter's ``_optim`` attribute which is later consumed
    by ``group_params()`` to create optimizer parameter groups.
    """
    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs
