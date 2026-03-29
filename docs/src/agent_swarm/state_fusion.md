# RWKV State Fusion

The `RWKVStateFusion` module combines recurrent states from multiple specialist agents into a single fused state for the primary agent. Implementation is in `src/aegir/swarm/state_fusion.py`.

## Input Format

Each agent produces a per-layer recurrent state tensor of shape:

```
(B, H, K, V)
```

where `B` is batch size, `H = num_heads`, and `K = V = head_size`. Given `N` agents, the fusion module receives a list of `N` such tensors and outputs a single tensor of the same shape.

Internally, the input list is stacked into a single tensor of shape `(B, N, H, K, V)`.

## Fusion Modes

### `weighted_sum` -- Attention Over Agent States

Uses a learnable query vector per head and a key projection to compute attention weights over agents.

**Parameters:**
- `query`: `(H, K)` -- learnable query per attention head
- `key_proj`: Linear mapping `K*V -> K` (no bias)

**Computation:**

```
flat   = reshape(stacked, [B, N, H, K*V])
keys   = key_proj(flat)                      # (B, N, H, K)
scores = einsum("bnhk, hk -> bnh", keys, query)
weights = softmax(scores, dim=1)             # (B, N, H)
fused  = einsum("bnh, bnhkv -> bhkv", weights, stacked)
```

Each head independently learns which agents to attend to. This is the default mode and generally the most effective, since it allows fine-grained per-head routing without excessive parameters.

### `gated` -- Learnable Per-Agent Gates

A simpler approach with a single learnable gate vector.

**Parameters:**
- `gates`: `(N,)` -- initialized to `1/N` (uniform)

**Computation:**

```
weights = softmax(gates, dim=0)   # (N,)
fused   = einsum("n, bnhkv -> bhkv", weights, stacked)
```

All heads share the same agent weighting. This is cheaper than `weighted_sum` but less expressive -- it cannot learn head-specific preferences for different specialists.

### `concat_project` -- Concatenate and Project

The most expressive mode. Concatenates all agent states along the agent dimension and projects back.

**Parameters:**
- `proj`: Linear mapping `N*K*V -> K*V` (no bias)

**Computation:**

```
flat      = reshape(permute(stacked, [0,2,1,3,4]), [B, H, N*K*V])
projected = proj(flat)           # (B, H, K*V)
fused     = reshape(projected, [B, H, K, V])
```

This allows arbitrary mixing of information across agents within each head but scales linearly in parameters with the number of agents.

## Usage Example

```python
from aegir.swarm.state_fusion import RWKVStateFusion

fusion = RWKVStateFusion(
    num_heads=8,
    head_size=64,
    num_agents=3,
    mode="weighted_sum",
)

# agent_states: list of 3 tensors, each (B, 8, 64, 64)
fused_state = fusion(agent_states)  # (B, 8, 64, 64)
```

## Mode Selection Guidelines

| Mode | Parameters | Per-head routing | Best for |
|------|-----------|-----------------|----------|
| `weighted_sum` | `O(H*K + K*V*K)` | Yes | General use, default |
| `gated` | `O(N)` | No | Quick experiments, few agents |
| `concat_project` | `O(N*K*V*K*V)` | Yes | Maximum expressiveness, small N |
