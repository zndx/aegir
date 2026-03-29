"""
Cross-agent state space projection for RWKV recurrent states.

Aligns recurrent states between agents with potentially different
architectures or training histories. Inspired by LatentMAS (arXiv:2511.20639)
W_a alignment matrices, adapted for RWKV's matrix-valued recurrent states.

For matrix states (att_kv): bilinear projection S' = W_l @ S @ W_r^T
For vector states (att_x_prev, ffn_x_prev): linear projection x' = W @ x
"""

import torch
import torch.nn as nn


class AlignmentProjection(nn.Module):
    """Projects recurrent states between agents of potentially different sizes.

    Handles both matrix states (att_kv) and vector states (att_x_prev,
    ffn_x_prev) with appropriate projection types.

    Args:
        source_num_heads: Number of heads in source agent.
        source_head_size: Head size of source agent.
        target_num_heads: Number of heads in target agent.
        target_head_size: Head size of target agent.
        source_d_model: Source hidden dimension (for vector states).
        target_d_model: Target hidden dimension (for vector states).
    """

    def __init__(
        self,
        source_num_heads: int,
        source_head_size: int,
        target_num_heads: int,
        target_head_size: int,
        source_d_model: int = None,
        target_d_model: int = None,
    ):
        super().__init__()
        self.source_num_heads = source_num_heads
        self.source_head_size = source_head_size
        self.target_num_heads = target_num_heads
        self.target_head_size = target_head_size

        source_d = source_d_model or source_num_heads * source_head_size
        target_d = target_d_model or target_num_heads * target_head_size

        # Matrix state projection: S' = W_l @ S @ W_r^T
        # When head counts differ, we flatten and reshape
        self.needs_matrix_proj = (
            source_num_heads != target_num_heads
            or source_head_size != target_head_size
        )
        if self.needs_matrix_proj:
            source_flat = source_num_heads * source_head_size * source_head_size
            target_flat = target_num_heads * target_head_size * target_head_size
            self.matrix_proj = nn.Linear(source_flat, target_flat, bias=False)

        # Vector state projection
        self.needs_vector_proj = source_d != target_d
        if self.needs_vector_proj:
            self.vector_proj = nn.Linear(source_d, target_d, bias=False)

    def forward_matrix(self, state: torch.Tensor) -> torch.Tensor:
        """Project matrix recurrent state (att_kv).

        Args:
            state: (B, H_s, K_s, V_s) source recurrent state.

        Returns:
            (B, H_t, K_t, V_t) projected state.
        """
        if not self.needs_matrix_proj:
            return state

        B = state.shape[0]
        flat = state.reshape(B, -1)
        projected = self.matrix_proj(flat)
        return projected.view(
            B, self.target_num_heads, self.target_head_size, self.target_head_size
        )

    def forward_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Project vector state (att_x_prev or ffn_x_prev).

        Args:
            x: (B, D_source) vector state.

        Returns:
            (B, D_target) projected state.
        """
        if not self.needs_vector_proj:
            return x
        return self.vector_proj(x)
