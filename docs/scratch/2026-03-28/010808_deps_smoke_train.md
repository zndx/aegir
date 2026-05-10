# Dependencies, Smoke Test, and Training Script

## Summary

Resolved CUDA dependency build issues (CXX11 ABI mismatch with nix environment), fixed several first-pass implementation bugs, got smoke tests passing, and created a training script with successful smoke training run.

## Dependency Resolution

### Problem
The nix/devenv environment uses GCC 15, which is too new for CUDA 12.4. System GCC 11 works but the nix environment injects flags that cause CXX11 ABI mismatch with torch's cu124 wheels (torch uses `_GLIBCXX_USE_CXX11_ABI=0`).

Additionally, both mamba-ssm and flash-attn use a `CachedWheelsCommand` pattern that tries to download prebuilt wheels from GitHub releases before building from source. These prebuilt wheels may have incorrect ABI.

### Solution
1. Use `env -i` to get a clean build environment (strip nix vars)
2. Patch `setup.py` for both mamba-ssm and flash-attn to explicitly add `_abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=0"` to both CXX and NVCC compile args
3. Set `MAMBA_FORCE_BUILD=TRUE` / `FLASH_ATTENTION_FORCE_BUILD=TRUE` to skip prebuilt wheel downloads

### Result
- causal-conv1d 1.6.1: imports OK
- mamba-ssm 2.3.1: imports OK (built from patched source)
- flash-attn 2.8.3: rebuilding from patched source (still compiling)

## Code Fixes

### `modules/utils.py` — `get_stage_cfg()`
Empty lists (e.g., `window_size=[]`) caused `IndexError` when indexing by stage. Fixed to return `None` for out-of-range indices.

### `modules/block.py` — `LayerNormPrenorm`
`nn.LayerNorm` doesn't support the `residual`/`prenorm`/`residual_in_fp32` kwargs that flash-attn's `RMSNorm` provides. Created `LayerNormPrenorm` wrapper that implements the prenorm residual interface.

### `modules/isotropic.py` — final norm
Same `LayerNormPrenorm` fallback applied to the final `rmsnorm` in `Isotropic`.

### `modules/rosa.py` — `_RosaQKV1BitOp.backward()`
The ROSA autograd function had no `backward()` method. Added a no-op backward that passes zero gradients for Q/K/V (suffix automaton is non-differentiable) and passes through gradients for the learnable `emb` parameter.

### `utils/train.py` — `group_params()`
Parameters without `_optim` attribute caused `AttributeError`. Added initialization of `_optim={}` for all params before grouping.

## Smoke Test Results

Both model heads pass forward + backward on CUDA:
- `AegirForCausalLM`: logits (2, 64, 65536), 2 boundary prediction stages, 46.8M params (tiny config)
- `AegirForColumnAnnotation`: logits (2, 91), 2 boundary prediction stages, 11.8M params (tiny config)

## Training Script

Created `train.py` with:
- 3 model sizes: tiny (11.8M), small, base
- Synthetic data mode for smoke testing (`--smoke-test`)
- DDP multi-GPU support (`torchrun --nproc_per_node=6`)
- Mixed precision (bfloat16 autocast)
- Cosine LR schedule with warmup
- Load balancing loss for routing regularization
- CTA (CrossEntropyLoss) / CPA (BCEWithLogitsLoss)
- Validation + best model checkpointing by macro F1

### Training Smoke Run (3 epochs, tiny model, synthetic data)
```
Epoch   1/3 | train loss=4.7376 (task=4.6315 lb=1.0611) F1=0.000/0.000 | val loss=4.6174 | 48.3s
Epoch   2/3 | train loss=4.7206 (task=4.6205 lb=1.0003) F1=0.008/0.001 | val loss=4.4971 | 15.6s
Epoch   3/3 | train loss=4.6449 (task=4.5449 lb=1.0002) F1=0.008/0.000 | val loss=4.4971 | 15.4s
```
Loss decreasing. First epoch slower due to CUDA kernel compilation caching.
