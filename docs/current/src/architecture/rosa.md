# ROSA Suffix Automaton

ROSA (RWKV Online Suffix Automaton) provides lossless infinite-range exact sequence matching as a complement to RWKV-7's learned recurrent processing. While RWKV-7 maintains a compressed state that approximates the input history, ROSA can retrieve exact substring matches from arbitrarily far in the past.

Reference: "ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching" (arXiv:2602.02499), ported from RWKV-v8 (BlinkDL/RWKV-LM).

## Algorithm Overview

ROSA constructs an online suffix automaton over discretized hidden representations. For each position in the query sequence, it finds the longest suffix of the query that appears as a substring in the key sequence seen so far, then returns the corresponding value from the position immediately after the match.

The core operation is `rosa_qkv_ref(qqq, kkk, vvv)`:

1. Maintain an online suffix automaton built incrementally from the key sequence.
2. For each new position `i`:
   - **Query phase**: Walk the automaton to find the longest suffix of `qqq[:i+1]` that matches a substring in `kkk[:i]`.
   - **Key phase**: Extend the automaton with `kkk[i]`.
3. If a match of sufficient length is found, return `vvv[match_end + 1]`. Otherwise return a sentinel value.

The suffix automaton provides O(n) construction and O(n) total query time, making the entire operation linear in sequence length.

## 1-Bit Binarization

To convert continuous hidden states into discrete tokens suitable for suffix automaton matching, ROSA uses 1-bit binarization:

```
x_binary = (x > 0) ? 1 : 0
```

This is applied **per channel** across the hidden dimension. Given a hidden state tensor of shape `(B, T, C)`:

1. **Binarize**: `q_bin[b, t, c] = uint8(q[b, t, c] > 0)` (same for k, v).
2. **Transpose**: Reshape from `(B, T, C)` to `(B*C, T)` -- each channel becomes an independent sequence.
3. **Match**: Run `rosa_qkv_batch_ref` over all `B*C` channel sequences in parallel.
4. **Reconstruct**: Reshape indices back to `(B, T, C)`.
5. **Scale**: Output `= (2 * idx_float - 1) * emb`, where `emb` is a learnable `(1, 1, C)` scale parameter.

The matched bit value `1` maps to `+emb` and `0` maps to `-emb`, giving the output the same sign structure as the matched hidden representation scaled by a learnable magnitude.

## The `_RosaQKV1BitOp` Autograd Function

ROSA's suffix automaton is non-differentiable (it involves discrete automaton state transitions). The autograd function handles this:

- **Forward**: Binarize inputs, run suffix matching on CPU, scale by `emb`.
- **Backward**: Gradients for `q`, `k`, `v` are `None` (zero). Gradients for `emb` are passed through directly.

This means ROSA layers learn only through:
1. The learnable `emb` scale parameter.
2. The Q/K/V linear projections preceding ROSA (which receive gradients from other paths in the network through residual connections).
3. The surrounding block's residual connection.

The projections learn to produce hidden representations whose binarization yields useful matching patterns, even though the binarization itself has no gradient.

## CPU Execution

The suffix automaton runs on CPU. Tensors are moved to CPU before matching and results are moved back to the accelerator. This is a deliberate design choice:

- Suffix automata use pointer-chasing data structures (dictionaries, linked suffix links) that are not amenable to GPU parallelism.
- The per-channel parallelism (`B*C` independent sequences) provides sufficient throughput for moderate batch sizes.
- During inference, ROSA blocks primarily contribute during prefill; the `step` method falls back to zero output since the automaton requires the full sequence context.

## When to Use ROSA vs RWKV-7

| Use Case | ROSA (`r`/`R`) | RWKV-7 (`w`/`W`) |
|----------|----------------|-------------------|
| Exact pattern retrieval | Yes -- lossless via suffix matching | No -- compressed into finite state |
| Learned sequence processing | Limited -- only `emb` is trained | Full -- all parameters are trained |
| Inference (autoregressive) | Degrades (needs full context) | Efficient (O(1) state update) |
| Long-range dependencies | Infinite range, exact | Finite effective range, approximate |
| Training speed | Slower (CPU automaton) | Fast (Triton chunk kernel) |

In practice, ROSA blocks are best used sparingly alongside RWKV-7 blocks. A typical layout might be `"w4r1"` -- four RWKV-7 blocks for general sequence processing, one ROSA block for exact retrieval. The ROSA block acts as a "lookup table" that can surface exact matches from the input, while RWKV-7 handles the bulk of learned representation building.

## RWKV_ROSA Module

The `RWKV_ROSA` module wraps the ROSA matching in a standard time-mixing interface:

1. **Time-shift mixing**: Mix current token with previous token via learned interpolation (same as RWKV-7 but with only q/k/v coefficients).
2. **Q/K/V projection**: Linear projections from the mixed hidden states.
3. **ROSA matching**: `RosaQKV1Bit` on the projected q, k, v.
4. **Output projection**: Linear projection back to `d_model`.

The module is paired with either `RWKV_CMix` (relu^2 FFN, block code `r`) or `SwiGLU` (block code `R`) as its feedforward component.
