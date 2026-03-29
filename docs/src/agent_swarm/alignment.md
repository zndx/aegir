# LatentMAS Alignment Projection

The `AlignmentProjection` module maps recurrent states between agents that may have different architectures (different `d_model`, `num_heads`, or `head_size`). Implementation is in `src/aegir/swarm/alignment.py`.

## Problem

When fusing states from multiple agents, all states must share the same `(H, K, V)` dimensions. But specialists may have been trained with different model sizes. A CTA specialist with `d_model=256` and a CPA specialist with `d_model=512` produce incompatible recurrent states. The alignment projection resolves this mismatch.

## State Types

RWKV recurrent states consist of two kinds of tensors:

### Matrix States (`att_kv`)

The core recurrent state from time mixing. Shape: `(B, H, K, V)` where `K = V = head_size`.

**Projection:** When source and target have different `num_heads` or `head_size`, the matrix state is flattened and linearly projected:

```
S_flat = reshape(S_source, [B, H_s * K_s * V_s])
S_target = W_matrix @ S_flat
S_out = reshape(S_target, [B, H_t, K_t, V_t])
```

where `W_matrix` has shape `(H_t * K_t * V_t, H_s * K_s * V_s)`.

The LatentMAS paper (arXiv:2511.20639) proposes using bilinear projection `S' = W_l @ S @ W_r^T` and computing `W_a` via ridge regression on paired agent activations. Aegir instead trains the projection end-to-end as part of the swarm's gradient flow, which avoids the need for a separate alignment data collection phase and allows the projection to co-adapt with the fusion module.

### Vector States (`att_x_prev`, `ffn_x_prev`)

The previous-timestep hidden state cache used by RWKV's time-shift mechanism. Shape: `(B, D)` where `D = d_model`.

**Projection:** Simple linear mapping when `d_model` differs:

```
x_target = W_vector @ x_source
```

where `W_vector` has shape `(D_target, D_source)`.

## When Projections Are Needed

The module detects whether projection is needed at initialization:

```python
# Matrix projection: needed when head geometry differs
needs_matrix_proj = (
    source_num_heads != target_num_heads
    or source_head_size != target_head_size
)

# Vector projection: needed when d_model differs
needs_vector_proj = (source_d_model != target_d_model)
```

When source and target share the same architecture, both projections are identity operations (no parameters allocated).

## Usage

```python
from aegir.swarm.alignment import AlignmentProjection

align = AlignmentProjection(
    source_num_heads=4,   source_head_size=64,
    target_num_heads=8,   target_head_size=64,
    source_d_model=256,
    target_d_model=512,
)

# Project matrix state
att_kv_target = align.forward_matrix(att_kv_source)   # (B,4,64,64) -> (B,8,64,64)

# Project vector state
x_prev_target = align.forward_vector(x_prev_source)   # (B,256) -> (B,512)
```

## LatentMAS vs Aegir Approach

| Aspect | LatentMAS | Aegir |
|--------|-----------|-------|
| Alignment method | Ridge regression on collected pairs | End-to-end gradient training |
| Training data | Requires parallel agent runs | Learned during swarm training |
| Adaptability | Fixed after alignment phase | Continuously adapts |
| Projection type | Bilinear `W_l @ S @ W_r^T` | Flatten + linear (equivalent expressiveness) |

The end-to-end approach is viable because Aegir's swarm training already has gradient flow through the fusion module. The alignment projection sits in that gradient path and receives signal from the downstream task loss.
