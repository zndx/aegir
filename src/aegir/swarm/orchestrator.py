"""
K2.5 PARL-inspired swarm orchestrator.

Coordinates a primary trainable Aegir agent with multiple frozen specialist
agents. The orchestrator decides when to activate specialists, fuses their
recurrent states into the primary agent's state, and manages the PARL
reward structure.

Reference: Kimi K2.5 (arXiv:2602.02276) — Parallel Agent Reinforcement Learning
"""

from typing import Optional

import torch
import torch.nn as nn

from aegir.swarm.state_fusion import RWKVStateFusion
from aegir.swarm.specialist import FrozenSpecialist


class SpecialistRouter(nn.Module):
    """Routes inputs to specialists based on learned routing scores.

    Args:
        d_model: Hidden dimension of the primary agent.
        num_specialists: Number of available specialists.
    """

    def __init__(self, d_model: int, num_specialists: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_specialists, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute specialist activation scores.

        Args:
            hidden_states: (B, D) primary agent's hidden representation
                (e.g., pooled from first routing layer).

        Returns:
            (B, num_specialists) activation scores (sigmoid-gated).
        """
        return torch.sigmoid(self.router(hidden_states))


class SwarmOrchestrator(nn.Module):
    """PARL orchestrator for multi-agent column annotation.

    Holds a primary Aegir model (trainable) and manages frozen specialist
    agents. The orchestrator:
    1. Routes inputs to relevant specialists via learned gating
    2. Fuses specialist recurrent states into the primary's state
    3. Supports PARL reward structure for RL post-training

    PARL reward components (for future RL training):
        r_perf: F1 accuracy on annotation tasks
        r_parallel: Load balancing + specialist utilization
        r_finish: Completion quality (all columns annotated)
        Combined: r = λ₁·r_parallel + λ₂·r_finish + r_perf

    Args:
        primary_model: Trainable primary Aegir model.
        specialists: List of frozen specialist agents.
        fusion: State fusion module for combining specialist states.
        activation_threshold: Minimum routing score to activate a specialist.
    """

    def __init__(
        self,
        primary_model: nn.Module,
        specialists: list[FrozenSpecialist],
        fusion: RWKVStateFusion,
        d_model: int,
        activation_threshold: float = 0.5,
    ):
        super().__init__()
        self.primary = primary_model
        self.specialists = nn.ModuleList(specialists)
        self.fusion = fusion
        self.router = SpecialistRouter(d_model, len(specialists))
        self.activation_threshold = activation_threshold

        # PARL reward weights (for RL training — not used in supervised phase)
        self.lambda_parallel = 0.3
        self.lambda_finish = 0.1

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        routing_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with optional specialist activation.

        Args:
            input_ids: (B, L) input token IDs.
            mask: (B, L) attention mask.
            routing_hidden: (B, D) hidden states for specialist routing.
                If None, specialist activation is skipped.
            **kwargs: Additional model arguments.

        Returns:
            Dict with primary output and specialist activation info.
        """
        specialist_outputs = []
        activation_mask = None

        if routing_hidden is not None and len(self.specialists) > 0:
            # Compute routing scores
            scores = self.router(routing_hidden)  # (B, num_specialists)
            activation_mask = scores > self.activation_threshold

            # Run activated specialists
            for i, specialist in enumerate(self.specialists):
                if activation_mask[:, i].any():
                    result = specialist.process_and_extract_state(
                        input_ids, mask=mask, **kwargs
                    )
                    specialist_outputs.append(result)

        # Primary forward pass
        primary_output = self.primary(input_ids, mask=mask, **kwargs)

        return {
            "output": primary_output,
            "specialist_outputs": specialist_outputs,
            "activation_mask": activation_mask,
        }

    def compute_parl_reward(
        self,
        r_perf: torch.Tensor,
        r_parallel: torch.Tensor,
        r_finish: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PARL combined reward.

        Args:
            r_perf: Performance reward (F1 accuracy).
            r_parallel: Parallelism reward (specialist utilization).
            r_finish: Completion reward (annotation coverage).

        Returns:
            Combined reward scalar.
        """
        return (
            r_perf
            + self.lambda_parallel * r_parallel
            + self.lambda_finish * r_finish
        )
