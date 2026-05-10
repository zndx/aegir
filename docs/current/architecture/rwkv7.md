# RWKV-7 Time Mixing

RWKV-7 time mixing is the primary sequence processing mechanism in Aegir. It implements a linear recurrence with a matrix-valued state, combining the training efficiency of chunk-parallel computation with the inference efficiency of constant-memory recurrence. The implementation uses flash-linear-attention's optimized Triton kernels.

Reference: RWKV-v8 "Heron" (BlinkDL/RWKV-LM), fla `RWKV7Attention`.

## Core Recurrence

The recurrent state `S[t]` is a matrix of shape `(H, head_size, head_size)` per batch element, where `H` is the number of attention heads. The state update at each time step is:

```
S[t] = diag(w[t]) * S[t-1] + S[t-1] @ ab[t] + v[t] @ k[t]^T
```

where:
- `diag(w[t])` is the per-element exponential decay applied column-wise
- `ab[t] = (-kk[t])^T @ (kk[t] * a[t])^T` is the attention gate correction
- `v[t] @ k[t]^T` is the new key-value outer product

The output is read from the state via:

```
o[t] = S[t] @ r[t]
```

where `r[t]` is the receptance (query) vector.

## Time-Shift Mixing

Before computing projections, RWKV-7 mixes each token with its predecessor via learned interpolation coefficients. Given input `x[t]`:

```
delta[t] = x[t-1] - x[t]       (delta[0] = -x[0])

xr = x + delta * mu_r
xw = x + delta * mu_w
xk = x + delta * mu_k
xv = x + delta * mu_v
xa = x + delta * mu_a
xg = x + delta * mu_g
```

Each `mu_*` is a learnable `(1, 1, D)` parameter initialized with a position-and-layer-dependent schedule. This provides a simple form of local context mixing before the main recurrence.

## Decay LoRA

The decay vector `w[t]` controls how quickly the recurrent state forgets. It is computed via a low-rank adaptation:

```
w[t] = -softplus(-(w0 + tanh(W1 @ xw[t]) @ W2)) - 0.5
```

where:
- `w0` is a `(D,)` bias initialized with a position-dependent schedule
- `W1` is `(D, decay_low_rank_dim)` and `W2` is `(decay_low_rank_dim, D)`
- The result is in log-space (negative values); the `-0.5` ensures minimum decay

For the chunked training kernel (`chunk_rwkv7`), `w` is passed in log-space. For the single-token step, it is converted to the multiplicative factor:

```
w_step = exp(-0.606531 * sigmoid(w0 + tanh(W1 @ xw) @ W2))
```

## Attention Gate LoRA

The attention gate `a[t]` modulates the key's influence on the state update. It controls the `ab` correction term:

```
a[t] = sigmoid(a0 + A2(A1(xa[t])))
```

where `a0` is a `(D,)` bias and `A1`, `A2` form a low-rank bottleneck. The key is then modified as:

```
k'[t] = k[t] * (1 + (a[t] - 1) * k_a)
```

where `k_a` is a learnable per-dimension scale (initialized to 1.0).

## Value-First Sharing

RWKV-7 shares value information across layers via a "value-first" mechanism:

- **Layer 0**: Stores its value projection as `v_first`.
- **Layers 1+**: Lerp their value toward `v_first`:

```
v[t] = v[t] + (v_first[t] - v[t]) * sigmoid(v0 + V2(V1(xv[t])))
```

This provides a residual-like connection specifically for value information, allowing deeper layers to reference the original value representation from layer 0.

## L2 Key Normalization

Keys are L2-normalized per head before entering the suffix automaton correction:

```
kk[t] = L2_normalize(k[t] * k_k)   per head
```

where `k_k` is a learnable per-dimension scale (initialized to 0.85). The normalized keys `kk` are used in the `ab` correction term but not in the main key-value outer product.

## Bonus Term

A direct key-query interaction term is added to the output:

```
bonus[t] = sum(r[t] * k[t] * r_k, dim=-1, keepdim=True) * v[t]
```

where `r_k` is a `(H, head_size)` parameter initialized with small random values. This provides a shortcut path that bypasses the recurrent state entirely.

## GroupNorm Output

The recurrent output is passed through GroupNorm (one group per attention head) before the bonus term is added:

```
o = GroupNorm(S[t] @ r[t])  +  bonus[t]
```

## Output Gating

The final output is gated via another LoRA:

```
g[t] = G2(sigmoid(G1(xg[t])))
output = o * g
output = W_o @ output
```

The output projection `W_o` is initialized to zero so that at initialization, RWKV-7 blocks contribute nothing to the residual stream.

## Training: Chunk-Parallel Computation

During training, the `chunk_rwkv7` kernel from flash-linear-attention processes the sequence in parallel chunks while maintaining exact recurrent semantics. The function signature:

```python
o, final_state = chunk_rwkv7(
    r, w, k, v,
    -kk, kk * a,            # ab decomposed as two rank-1 terms
    initial_state=state,     # (B, H, K, K) or None
    output_final_state=True,
)
```

Inputs are shaped `(B, T, H, head_size)` and `w` is in log-space.

## Inference: Token-by-Token Recurrence

During autoregressive inference, the `step` method implements the exact recurrence manually:

```python
vk = v @ k^T                    # (B, H, N, N)
ab = (-kk)^T @ (kk * a)^T       # (B, H, N, N)
S  = S * diag(w) + S @ ab + vk  # state update
o  = S @ r                       # read output
```

The recurrent state `S` is stored in `inference_params.key_value_memory_dict[layer_idx].att_kv` as a `float32` tensor of shape `(B, H, head_size, head_size)`.

## LoRA Dimension Auto-Calculation

If not explicitly specified in `RWKVConfig`, LoRA dimensions are computed from `d_model` following the fla convention:

```python
factor = head_size / 64
sqrt_d = sqrt(d_model)

decay_low_rank_dim = max(32, round(2.5 * sqrt_d * factor / 32) * 32)
gate_low_rank_dim  = max(32, round(5.0 * sqrt_d / 32) * 32)
a_low_rank_dim     = max(32, round(2.5 * sqrt_d * factor / 32) * 32)
v_low_rank_dim     = max(32, round(1.7 * sqrt_d * factor / 32) * 32)
```

All dimensions are rounded up to multiples of 32 for hardware efficiency.

## Weight Initialization

Initialization follows RWKV-7 conventions with layer-dependent schedules:

- **Time-shift coefficients** (`mu_*`): Initialized as `1 - d^(c * ratio)` where `d` is a per-dimension ramp `[0, 1)`, `c` is a coefficient specific to each mix type, and `ratio` varies from 1 (first layer) to 0 (last layer).
- **Decay bias** (`w0`): Initialized as `-7 + 5 * (d / D)^(0.85 + ratio^0.5)`, giving a range from fast decay (early dimensions) to slow decay (late dimensions).
- **Key normalization** (`k_k`): 0.85 uniformly.
- **Key attention scale** (`k_a`): 1.0 uniformly.
- **Bonus** (`r_k`): Small random normal (std=0.1).
- **Output projection** (`W_o`): Zero initialized.
