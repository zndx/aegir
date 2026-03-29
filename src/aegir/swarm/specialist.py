"""
Frozen specialist agent wrapper.

Wraps an Aegir model with frozen parameters and provides an interface
for running forward passes and extracting recurrent states for fusion
with the primary agent.
"""

from typing import Optional

import torch
import torch.nn as nn

from aegir.swarm.alignment import AlignmentProjection


class FrozenSpecialist(nn.Module):
    """Wraps a pre-trained Aegir model as a frozen specialist.

    The specialist runs forward passes with no gradient computation and
    exposes its recurrent states for fusion with the primary agent. An
    optional AlignmentProjection handles cross-agent state mapping when
    the specialist and primary have different architectures.

    Args:
        model: Pre-trained Aegir model (will be frozen).
        alignment: Optional projection for cross-agent state alignment.
        specialist_id: Identifier for this specialist.
    """

    def __init__(
        self,
        model: nn.Module,
        alignment: Optional[AlignmentProjection] = None,
        specialist_id: str = "specialist_0",
    ):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.alignment = alignment
        self.specialist_id = specialist_id

    @torch.no_grad()
    def process_and_extract_state(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """Run forward pass and extract recurrent states.

        Args:
            input_ids: (B, L) input token IDs.
            mask: (B, L) attention mask.
            **kwargs: Additional model arguments.

        Returns:
            Dict with:
                - ``output``: Model output (logits, etc.)
                - ``states``: List of per-layer RWKVBlockState or equivalent,
                    optionally projected through alignment.
        """
        output = self.model(input_ids, mask=mask, **kwargs)

        # Extract recurrent states from the backbone
        states = self._extract_states()

        # Apply alignment projection if configured
        if self.alignment is not None:
            states = self._align_states(states)

        return {
            "output": output,
            "states": states,
            "specialist_id": self.specialist_id,
        }

    def _extract_states(self) -> list:
        """Extract recurrent states from the model's backbone.

        This traverses the Aegir hierarchy to find RWKV block states.
        Currently returns an empty list — will be connected to the backbone's
        state extraction once inference cache is populated during forward.
        """
        # TODO: Connect to backbone inference cache or add state extraction
        # hooks. For now, return placeholder for scaffolding.
        return []

    def _align_states(self, states: list) -> list:
        """Apply alignment projection to extracted states."""
        if self.alignment is None:
            return states

        aligned = []
        for state in states:
            aligned_state = {}
            if hasattr(state, "att_kv") and state.att_kv is not None:
                aligned_state["att_kv"] = self.alignment.forward_matrix(state.att_kv)
            if hasattr(state, "att_x_prev"):
                aligned_state["att_x_prev"] = self.alignment.forward_vector(
                    state.att_x_prev
                )
            if hasattr(state, "ffn_x_prev") and state.ffn_x_prev is not None:
                aligned_state["ffn_x_prev"] = self.alignment.forward_vector(
                    state.ffn_x_prev
                )
            aligned.append(aligned_state)

        return aligned
