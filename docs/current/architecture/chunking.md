# Hierarchical Dynamic Chunking

Dynamic chunking is Aegir's mechanism for content-dependent hierarchical segmentation. Rather than using a fixed tokenizer, the model learns to predict chunk boundaries based on the hidden representations themselves. This module is adapted from H-Net (goombalab/hnet).

## Overview

The chunking pipeline has three components that work together at each non-innermost stage of the hierarchy:

1. **RoutingModule** -- predicts which tokens are chunk boundaries
2. **ChunkLayer** -- downsamples the sequence by selecting boundary tokens
3. **DeChunkLayer** -- reconstructs the full-length sequence from chunk outputs via EMA

## RoutingModule: Boundary Prediction

The routing module decides where to place chunk boundaries by measuring how different consecutive hidden states are.

### Algorithm

For a sequence of hidden states `h[0], h[1], ..., h[L-1]`:

1. Project consecutive pairs through learnable Q and K matrices (initialized to identity).
2. Compute cosine similarity between adjacent projected states:
   ```
   cos_sim[t] = cosine(Q @ h[t], K @ h[t+1])
   ```
3. Convert to boundary probability:
   ```
   p[t] = clamp((1 - cos_sim[t]) / 2, 0, 1)
   ```
4. The first token always gets `p = 1.0` (always a boundary).
5. Threshold at 0.5: if `p[t] > 0.5`, token `t` is a boundary.

High dissimilarity between consecutive states means the content is changing -- a natural place to start a new chunk. The Q/K projections are initialized to identity so the model starts with raw cosine similarity and can learn to refine the boundary criterion.

### Handling Variable-Length Sequences

The routing module supports two modes:

- **Padded mode** (`mask` provided): Standard `(B, L, D)` tensors with a boolean mask. Boundary predictions outside the mask are suppressed.
- **Packed mode** (`cu_seqlens` provided): Sequences concatenated into a single `(1, total_len, D)` tensor with cumulative sequence lengths. The first token of each sequence in the pack is forced to be a boundary.

## ChunkLayer: Downsampling

Once boundaries are predicted, `ChunkLayer` selects only the boundary tokens to form a shorter sequence.

In padded mode:
1. Count how many boundary tokens each batch element has.
2. Sort token indices so boundary tokens come first.
3. Gather the first `max_boundaries` tokens per batch element.
4. Produce a new mask indicating which positions in the shorter sequence are valid.

In packed mode:
1. Boolean-index the boundary tokens directly from the flat sequence.
2. Recompute `cu_seqlens` for the shorter packed sequence.

The output is a shorter sequence containing only the tokens that were at chunk boundaries.

## DeChunkLayer: Reconstruction via EMA

After the inner hierarchy processes the chunked (shorter) sequence, `DeChunkLayer` reconstructs the full-length sequence. The key insight is that non-boundary tokens should smoothly interpolate from their nearest preceding boundary token's output.

### EMA Scan

The reconstruction uses an exponential moving average (EMA) scan:

```
y[0] = x[0]
y[t] = decay[t] * y[t-1] + (1 - decay[t]) * x[t]
```

where `decay[t] = 1 - p[t]` and `p[t]` is the boundary probability for token `t`.

At boundary tokens (`p ~ 1`), the output snaps to the new chunk value. At non-boundary tokens (`p ~ 0`), the output carries forward the previous value. The boundary probability controls the blend continuously, allowing gradient flow through the routing decisions.

### Reconstruction Steps

1. Reorder the chunk outputs according to the original boundary positions.
2. Map each position in the full sequence to its cumulative boundary count (i.e., which chunk it belongs to).
3. Run the EMA scan over the reordered chunk outputs with boundary-probability-derived decay factors.
4. Gather the EMA outputs back to the original sequence positions.

## Residual Connection

The entire chunk/process/dechunk pipeline is wrapped in a residual connection:

```
output = dechunk_output * STE(selected_probs) + residual_proj(encoder_output)
```

The `residual_proj` is a linear layer initialized to zero, so at initialization the chunking pathway contributes nothing and the model starts as a simple encoder-decoder. The Straight-Through Estimator (STE) passes gradients through the discrete routing decisions.

## Recursive Nesting

The chunking pattern nests recursively. Consider a 3-stage hierarchy:

```
arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]
```

- **Stage 0**: Encode the full byte sequence, predict boundaries, chunk down, pass to Stage 1, dechunk back up, decode.
- **Stage 1**: Encode the chunked sequence from Stage 0, predict boundaries again on this shorter sequence, chunk down further, pass to Stage 2, dechunk, decode.
- **Stage 2**: Process the doubly-chunked sequence with a flat stack of blocks (no further chunking).

Each level of chunking reduces the sequence length by a data-dependent factor. For byte-level input, the first level might learn character-like boundaries; the second level might learn word-like or phrase-like boundaries. The model discovers its own hierarchy of tokenization.

## Inference: Token-by-Token Stepping

During autoregressive inference, each component has a `step` method for single-token processing:

- **RoutingModule.step**: Compares the new token against the previously seen token's hidden state. If the boundary probability exceeds 0.5, the token starts a new chunk.
- **ChunkLayer.step**: If the token is a boundary, pass it through to the inner hierarchy. Otherwise, skip the inner hierarchy entirely.
- **DeChunkLayer.step**: Blend the new chunk output (if any) with the previous EMA value using the boundary probability as the mixing weight.

This means that during inference, the inner hierarchy only runs when a chunk boundary is detected, saving compute on non-boundary tokens.
