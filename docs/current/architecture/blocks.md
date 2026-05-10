# Block Types Reference

Aegir's architecture is built from modular blocks, each consisting of a **mixer** (the sequence processing module) and an optional **MLP** (the feedforward network). Blocks are identified by single-character codes and composed into layout strings that define the architecture at each stage.

## Block Code Table

| Code | Mixer | MLP | Description |
|------|-------|-----|-------------|
| `w` | RWKV-7 TimeMix | CMix (relu^2) | Full RWKV-7 recurrence with RWKV-style channel mixing |
| `W` | RWKV-7 TimeMix | SwiGLU | Full RWKV-7 recurrence with SwiGLU feedforward |
| `r` | ROSA (suffix automaton) | CMix (relu^2) | Exact pattern matching with RWKV-style channel mixing |
| `R` | ROSA (suffix automaton) | SwiGLU | Exact pattern matching with SwiGLU feedforward |
| `t` | Multi-Head Attention | None | Causal MHA with no feedforward |
| `T` | Multi-Head Attention | SwiGLU | Standard transformer block |
| `m` | Mamba-2 (SSM) | None | State-space model with no feedforward |
| `M` | Mamba-2 (SSM) | SwiGLU | State-space model with SwiGLU feedforward |

### Convention

- **Lowercase** codes use RWKV-native FFN (CMix with relu^2) or no FFN at all.
- **Uppercase** codes use SwiGLU as the feedforward network.
- For `w`/`W` and `r`/`R`, lowercase uses CMix; uppercase uses SwiGLU.
- For `t`/`T` and `m`/`M`, lowercase has no MLP; uppercase adds SwiGLU.

## The Block Wrapper

Every block follows the **pre-norm residual pattern**:

```
                    +---> norm1 --> mixer ---+
                    |                       |
hidden_states ----->+                       +-----> hidden_states
(+ residual)        |                       |      (+ residual)
                    +---> norm2 --> mlp ----+  (if MLP exists)
```

Concretely, the `Block` class implements:

```python
# Mixer sub-block
hidden_states, residual = norm1(hidden_states, residual, prenorm=True)
hidden_states = mixer(hidden_states)

# MLP sub-block (if present)
hidden_states, residual = norm2(hidden_states, residual, prenorm=True)
hidden_states = mlp(hidden_states)
```

The pre-norm pattern accumulates the residual stream separately from the normalized hidden states. The normalization module (RMSNorm from flash-attn, or a LayerNorm fallback) handles residual accumulation internally when `prenorm=True`.

### Residual Height Counting

Each block contributes to the "height" of its parent `Isotropic` module, which is used for output projection scaling during initialization:

- Lowercase blocks (single residual addition): height += 1
- Uppercase blocks (mixer + MLP, two residual additions): height += 2

## MLP Variants

### CMix (RWKV Channel Mixing)

Used by lowercase RWKV codes (`w`, `r`). A simple feedforward with relu^2 activation:

```python
# Time-shift mixing
xx = time_shift(x) - x
k = x + xx * x_k

# Feedforward
k = relu(W_key @ k) ** 2    # D -> 4D, relu squared
output = W_value @ k          # 4D -> D
```

The expansion factor defaults to `rwkv_cfg.dim_ffn_mult` (default 4.0). CMix includes its own time-shift mixing, independent of the mixer's time-shift.

### SwiGLU

Used by uppercase codes (`W`, `R`, `T`, `M`). The standard SwiGLU feedforward (Shazeer 2020):

```python
y = W_fc1 @ x                # D -> 2 * D_intermediate
y, gate = split(y)           # Each D_intermediate
y = silu(gate) * y
output = W_fc2 @ y            # D_intermediate -> D
```

The intermediate dimension defaults to `8/3 * d_model`, rounded up to the nearest multiple of 128.

## Layout String Parsing

Architecture layout strings encode a sequence of block types and their counts. The string is parsed by the `Isotropic` module using a regex:

```python
re.findall(r"([mMtTrRwW])(\d+)", arch_layout)
```

Examples:

| Layout String | Parsed Blocks |
|---------------|---------------|
| `"w4"` | 4 RWKV-7+CMix blocks |
| `"w4T1r2"` | 4 RWKV-7+CMix, 1 MHA+SwiGLU, 2 ROSA+CMix |
| `"W8"` | 8 RWKV-7+SwiGLU blocks |
| `"m2w4m2"` | 2 Mamba-2, 4 RWKV-7+CMix, 2 Mamba-2 |

Within a layout string, blocks are instantiated in order with sequential `layer_idx` values. The total layer count across all block types in the string is used for RWKV-7's position-dependent weight initialization.

## The `create_block` Function

`create_block()` is the factory function that dispatches on the block code character:

```python
block = create_block(
    arch="w",                    # block code
    d_model=192,                 # hidden dimension
    d_intermediate=512,          # SwiGLU intermediate dim (for uppercase codes)
    ssm_cfg={...},               # Mamba-2 config (for m/M)
    attn_cfg={...},              # MHA config (for t/T)
    rwkv_cfg=RWKVConfig(...),    # RWKV config (for w/W/r/R)
    layer_idx=0,                 # layer index for cache keying
    num_hidden_layers=12,        # total layers for init scheduling
)
```

The function:

1. Selects the **mixer class** based on the code character.
2. Selects the **MLP class**: CMix for `w`/`r`, SwiGLU for uppercase, `nn.Identity` for `t`/`m`.
3. Selects the **normalization class**: flash-attn's RMSNorm if available, otherwise a LayerNorm fallback with prenorm support.
4. Constructs and returns a `Block` instance with the selected components.

## Value-First Sharing Across Blocks

When an `Isotropic` module contains RWKV-7 blocks (`w`/`W`), it maintains a shared `v_first = [None]` container. This mutable list is passed as a `mixer_kwarg` to every RWKV-7 block:

- The first RWKV-7 block (layer_idx 0 within the Isotropic) stores its value projection in `v_first[0]`.
- Subsequent RWKV-7 blocks lerp their value toward `v_first[0]` via a learnable gate.

This sharing is local to each `Isotropic` instance -- encoder, decoder, and main network at each stage each have their own `v_first` container.
