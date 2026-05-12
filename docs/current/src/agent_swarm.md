# Agent Swarm Architecture

The agent-swarm modules in `src/aegir/swarm/` are infrastructure for future multi-agent training. The system's current operational training pipeline (described in the [semantic-engine authoritative reference](./ontology/production_state.md)) is single-policy; this chapter documents the design of the swarm modules and the architectural rationale for landing them in the codebase ahead of an operational multi-agent task.

The swarm shares compact RWKV recurrent state tensors between agents rather than exchanging text messages or attention KV caches — a communication medium that is uniquely efficient for recurrent architectures.

## Why RWKV state sharing

The central insight is that RWKV's recurrent state is **constant in sequence length**. Each layer's state is a matrix of shape `(H, K, V)` where `H` is the number of heads and `K = V = head_size`. The total state size per layer is:

```
O(H * head_size^2) = O(d_model * head_size) = O(d^2)
```

This is independent of how many tokens the agent has processed.

For a swarm of `N` agents, the cost of sharing all recurrent states is:

```
RWKV:        O(N * d^2)          -- constant in sequence length
Transformer: O(N * n * d)        -- linear in sequence length n
```

At context lengths of 4k-128k tokens with typical `d = 512-4096`, RWKV state sharing is orders of magnitude cheaper. The LatentMAS paper (arXiv:2511.20639) quantifies this as **235-471x more information-dense** than text-based inter-agent communication, since the recurrent state encodes a compressed summary of the entire processing history.

## Swarm Components

The swarm consists of four modules:

| Module | File | Purpose |
|--------|------|---------|
| `RWKVStateFusion` | `src/aegir/swarm/state_fusion.py` | Combine N agent states into one |
| `AlignmentProjection` | `src/aegir/swarm/alignment.py` | Map states between different-sized agents |
| `FrozenSpecialist` | `src/aegir/swarm/specialist.py` | Wrap pre-trained models as frozen agents |
| `SwarmOrchestrator` | `src/aegir/swarm/orchestrator.py` | K2.5 PARL routing and reward |

## State Fusion Modes

`RWKVStateFusion` supports three strategies for combining agent states:

1. **`weighted_sum`** -- Attention-weighted combination using learnable query/key projections. The orchestrator learns which agents to trust per head.

2. **`gated`** -- Per-agent softmax gates. Simpler than attention but still differentiable. Good baseline for initial experiments.

3. **`concat_project`** -- Concatenate all agent states and project back to single-agent dimensions. Most expressive but `O(N)` in parameter count.

See [RWKV State Fusion](./agent_swarm/state_fusion.md) for mathematical details.

## Information Density Advantage

LatentMAS demonstrates that recurrent state communication dramatically outperforms text-based multi-agent protocols. The recurrent state is a lossy but highly compressed representation of the agent's entire context window. Sharing it is equivalent to sharing a continuous-valued "summary" that preserves the information most relevant to the model's computation, rather than forcing that information through a text bottleneck.

For Aegir's column annotation task, this means a specialist trained on (say) geographic column types can share its accumulated understanding of a table's structure through a single `(H, K, V)` tensor per layer, rather than generating and parsing natural language explanations.
