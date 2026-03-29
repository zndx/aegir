# All-RWKV Refactor + Agent Swarm Scaffold — Complete

## Summary

Completed full 5-phase plan: all-RWKV architecture refactor, agent swarm scaffold, mdbook documentation, and verification.

## Phase 1: RWKV-7 TimeMix Module

- Created `src/aegir/modules/rwkv7_tmix.py` — full RWKV-7 time mixing with fla's `chunk_rwkv7` kernel
- Updated `RWKVBlockState` with `att_kv` for recurrent KV matrix `(B, H, K, V)`
- Added LoRA dimension fields to `RWKVConfig`

### Key Implementation Details
- Token shift via `fla.modules.token_shift`
- Decay in log space: `w = -softplus(-(w0 + tanh(w1(xw)) @ w2)) - 0.5`
- Value-first sharing: layer 0 sets v_first, subsequent layers lerp
- L2 key normalization + bonus term `(r * k * r_k).sum() * v`
- GroupNorm output with gated projection

## Phase 2: All-RWKV Refactor

- Added `w`/`W` block types (RWKV-7 + CMix / SwiGLU)
- Made `m`/`M` (Mamba-2) optional with lazy import
- Replaced `mamba_chunk_scan_combined` EMA in DeChunkLayer with custom `_ema_scan`
- Moved mamba-ssm to optional dependencies
- Default architectures now use `w` blocks everywhere

## Phase 3: Agent Swarm Scaffold

- `src/aegir/swarm/state_fusion.py` — 3 fusion modes (weighted_sum, gated, concat_project)
- `src/aegir/swarm/alignment.py` — Cross-agent state projection (matrix and vector)
- `src/aegir/swarm/specialist.py` — Frozen specialist wrapper
- `src/aegir/swarm/orchestrator.py` — K2.5 PARL orchestration

## Phase 4: Documentation

- 17 mdbook pages covering architecture, agent swarm, K2.5 PARL roadmap, and development guide

## Phase 5: Verification

- Smoke tests pass (main.py): both CausalLM and ColumnAnnotation models
- Training smoke test passes: 3 epochs, loss decreasing (4.73 → 4.63), no NaN/errors

## Issues Encountered and Fixed

### 1. Triton dtype mismatch under AMP
**Error**: `AssertionError: Both operands must be same dtype. Got bf16 and fp32` in chunk_rwkv7 kernel.
**Fix**: Explicitly cast all kernel inputs to bf16 AND wrap `chunk_rwkv7` in `torch.amp.autocast("cuda", enabled=False)`.

### 2. Inplace modification during backward
**Error**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [CUDABFloat16Type [16, 128]]`
**Fix**: `_ema_scan` was using `y[:, t] = ...` inplace assignment on a tensor in the computation graph. Replaced with list accumulation + `torch.stack`:
```python
outputs = [x[:, 0]]
for t in range(1, x.shape[1]):
    outputs.append(decay[:, t] * outputs[-1] + (1 - decay[:, t]) * x[:, t])
return torch.stack(outputs, dim=1)
```

### 3. NVIDIA library symlinks
**Error**: `libcudnn.so.9 not found`, `libnccl.so.2 not found`
**Fix**: Symlinked all nvidia .so files from system python3.10 to devenv venv.

## Version
Package version bumped to 0.2.0.
