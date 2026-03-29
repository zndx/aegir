"""
RWKV recurrent state fusion for multi-agent collaboration.

Combines recurrent states from multiple specialist agents into a single
fused state that the primary agent can use. RWKV's constant-size recurrent
state (n_head, head_size, head_size) is O(d²) regardless of sequence length,
making it far more efficient for inter-agent communication than transformer
KV caches (O(n×d)).

Reference: LatentMAS (arXiv:2511.20639) adapted for RWKV recurrent states.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVStateFusion(nn.Module):
    """Fuses RWKV recurrent states from multiple agents.

    Supports three fusion modes:
    - ``weighted_sum``: Attention-weighted combination of agent states
    - ``gated``: Per-agent learnable gates
    - ``concat_project``: Concatenate along agent dim, project back

    Args:
        num_heads: Number of attention heads in the RWKV model.
        head_size: Per-head dimension.
        num_agents: Number of specialist agents to fuse.
        mode: Fusion strategy.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_agents: int,
        mode: Literal["weighted_sum", "gated", "concat_project"] = "weighted_sum",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_agents = num_agents
        self.mode = mode

        if mode == "weighted_sum":
            # Attention over agent states — learnable query per head
            self.query = nn.Parameter(torch.randn(num_heads, head_size))
            self.key_proj = nn.Linear(head_size * head_size, head_size, bias=False)
        elif mode == "gated":
            # Per-agent learnable gates
            self.gates = nn.Parameter(torch.ones(num_agents) / num_agents)
        elif mode == "concat_project":
            # Project concatenated agent states back to single-agent dim
            self.proj = nn.Linear(
                num_agents * head_size * head_size,
                head_size * head_size,
                bias=False,
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode!r}")

    def forward(
        self,
        agent_states: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse recurrent states from multiple agents.

        Args:
            agent_states: List of N tensors, each (B, H, K, V) where
                H=num_heads, K=head_size, V=head_size.

        Returns:
            Fused state (B, H, K, V).
        """
        assert len(agent_states) == self.num_agents
        B = agent_states[0].shape[0]
        H, K, V = self.num_heads, self.head_size, self.head_size

        # Stack: (B, N, H, K, V)
        stacked = torch.stack(agent_states, dim=1)

        if self.mode == "weighted_sum":
            # Flatten each state per head: (B, N, H, K*V)
            flat = stacked.view(B, self.num_agents, H, K * V)
            # Project to key space: (B, N, H, K)
            keys = self.key_proj(flat)
            # Attention scores: (B, N, H)
            scores = torch.einsum("bnhk,hk->bnh", keys, self.query)
            weights = F.softmax(scores, dim=1)  # (B, N, H)
            # Weighted combination: (B, H, K, V)
            fused = torch.einsum("bnh,bnhkv->bhkv", weights, stacked)

        elif self.mode == "gated":
            weights = torch.softmax(self.gates, dim=0)  # (N,)
            fused = torch.einsum("n,bnhkv->bhkv", weights, stacked)

        elif self.mode == "concat_project":
            # Flatten all agents: (B, H, N*K*V)
            flat = stacked.permute(0, 2, 1, 3, 4).reshape(B, H, -1)
            # Project: (B, H, K*V)
            projected = self.proj(flat)
            fused = projected.view(B, H, K, V)

        return fused
