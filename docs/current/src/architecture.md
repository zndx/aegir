# Architecture Overview

Aegir is a recursive hierarchical sequence model. At the top level, it processes raw byte sequences through nested stages of encoding, dynamic chunking, inner processing, dechunking, and decoding. Each stage can use a different hidden dimension and a different mix of block types.

## Recursive Hierarchy

The architecture is defined by a nested list called `arch_layout`. For example:

```python
arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]
d_model     = [128,   192,   192]
```

This defines three stages (depth 0, 1, 2):

| Stage | Role | Layout | Dimension |
|-------|------|--------|-----------|
| 0 | Outermost encoder/decoder | `"w2"` / `"w2"` | 128 |
| 1 | Middle encoder/decoder | `"w2"` / `"w2"` | 192 |
| 2 | Innermost (main) | `"w4"` | 192 |

At each non-innermost stage, the data flow is:

```d2
direction: right

input: Input\n(B, L, D)
encoder: Encoder\n(Isotropic)
routing: Routing\nModule
chunk: Chunk\nLayer
main: Main Network\n(Aegir recursive)
dechunk: DeChunk\nLayer
decoder: Decoder\n(Isotropic)
output: Output\n(B, L, D)

input -> encoder -> routing -> chunk -> main -> dechunk -> decoder -> output

residual: Residual + STE {
  style.stroke-dash: 5
}
encoder -> residual
residual -> dechunk: skip {
  style.stroke-dash: 5
}
```

At the innermost stage, only the main network runs (no chunking). The recursion bottoms out at a flat `Isotropic` block stack.

### Data Flow in Detail

1. **Encoder**: A flat stack of blocks (e.g., 2 RWKV-7 blocks) processes the full-resolution sequence.
2. **Routing**: `RoutingModule` predicts boundary probabilities via cosine similarity. Tokens at predicted boundaries are selected as chunk representatives.
3. **Chunk**: `ChunkLayer` downsamples by keeping only boundary tokens, producing a shorter sequence.
4. **Main network**: The shorter sequence is processed by the next hierarchy level -- which may itself contain encoding, chunking, and another level of recursion.
5. **Dechunk**: `DeChunkLayer` reconstructs the full-length sequence via an EMA scan, blending chunk outputs back into non-boundary positions.
6. **Residual**: A skip connection around the entire chunk/process/dechunk block, gated via straight-through estimation of the routing probabilities.
7. **Decoder**: Another flat stack of blocks processes the reconstructed sequence.

### Dimension Padding

When inner stages have a larger hidden dimension than outer stages, Aegir pads the input with a learnable vector (`pad_dimension`) on entry and slices it off on exit. This avoids linear projection overhead at every stage transition.

## Why All-RWKV

The primary design choice is to use RWKV-7 time mixing at all stages rather than transformers or pure SSMs. The motivation is threefold:

### 1. Uniform O(1) Recurrent State

Every RWKV-7 block maintains a recurrent state of shape `(B, H, head_size, head_size)`. This is constant regardless of sequence length. During autoregressive inference, each token step updates this matrix and reads from it in O(head_size^2) time per head.

### 2. Agent State Fusion

For the agent swarm architecture, specialist agents process the same input and produce recurrent states. These states must be combined. RWKV states are fixed-size matrices that live in a well-defined linear space, making fusion via weighted sum, gating, or projection algebraically natural. In contrast:

- Transformer KV caches are O(L * d) and grow with sequence length, making fusion combinatorially expensive.
- Mamba-2 states are smaller but have different algebraic structure (diagonal recurrence).

### 3. Chunk-Parallel Training

The `chunk_rwkv7` kernel from flash-linear-attention enables training with parallel chunk processing while maintaining exact recurrent semantics. This gives near-transformer training throughput with recurrent inference efficiency.

## Comparison Table

| Property | RWKV-7 (`w`/`W`) | Mamba-2 (`m`/`M`) | Transformer (`t`/`T`) |
|----------|-------------------|--------------------|-----------------------|
| Training kernel | `chunk_rwkv7` (Triton) | Mamba-2 SSD (CUDA) | Flash Attention 2 |
| Recurrent state | `(H, K, K)` matrix | `(H, d_state)` vector | None (KV cache) |
| Inference memory | O(d^2) constant | O(d * d_state) constant | O(L * d) linear |
| State fusibility | Natural (matrix sum) | Possible (vector sum) | Impractical |
| Exact retrieval | Via ROSA blocks | No | Via full attention |
| FFN pairing | CMix (relu^2) or SwiGLU | SwiGLU or none | SwiGLU or none |

In practice, RWKV-7 blocks (`w`/`W`) are the default at every stage and are the only block code used in any current `arch_layout` (`main.py`, `train.py` for tiny / small / base). Mamba-2 (`m`/`M`) is available as an optional dependency for ablation and hybrid configurations. ROSA (`r`/`R`) implements a RWKV-8 suffix automaton for exact substring retrieval; it is functional in the block factory but is not part of any current default arch_layout — available for hybrid configurations and ablations. MHA (`t`/`T`) codes are declared in the block factory but the implementing module is not currently shipped.
