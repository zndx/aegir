"""
Agent swarm infrastructure for Aegir.

Provides LatentMAS-style RWKV recurrent state fusion and K2.5 PARL
orchestration patterns for multi-agent column annotation.

Modules:
    state_fusion: RWKV recurrent state fusion (weighted_sum, gated, concat_project)
    alignment: Cross-agent state space projection (bilinear for matrices, linear for vectors)
    orchestrator: K2.5 PARL task decomposition and specialist routing
    specialist: Frozen specialist agent wrapper with state extraction
"""

from aegir.swarm.state_fusion import RWKVStateFusion
from aegir.swarm.alignment import AlignmentProjection
from aegir.swarm.orchestrator import SwarmOrchestrator
from aegir.swarm.specialist import FrozenSpecialist

__all__ = [
    "RWKVStateFusion",
    "AlignmentProjection",
    "SwarmOrchestrator",
    "FrozenSpecialist",
]
