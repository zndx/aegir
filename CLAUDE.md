# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aegir is a hierarchical sequence modeling system using an all-RWKV-7 architecture with H-Net-style dynamic chunking for adaptive segmentation. Primary use case is relational data warehouse metadata tagging (Column Type Annotation and Column Property Annotation for wide tables). Includes agent swarm scaffold (LatentMAS-style RWKV state fusion + K2.5 PARL orchestration). Pre-alpha stage (v0.2.0).

Reference papers in `ref/`: H-Net (Dynamic Chunking), Retrieve-and-Verify, ROSA-Tuning.

## Architecture

The model uses a recursive hierarchical structure defined by `arch_layout`, a nested list where each stage is `[encoder, main_network, decoder]` or `[main_network]` (innermost):

```python
arch_layout = ["w2", ["w2", ["w4"], "w2"], "w2"]
d_model     = [128,   192,   192]  # per-stage hidden dim (padded/sliced between stages)
```

Non-innermost stages flow: encoder (Isotropic) → RoutingModule → ChunkLayer → main_network (Aegir recursive) → DeChunkLayer → decoder (Isotropic), with an STE-gated residual skip around the chunk/main/dechunk block.

**Block type codes** (lowercase = no SwiGLU MLP, uppercase = + SwiGLU; lowercase RWKV gets CMix relu² FFN):

| Code | Mixer | FFN | Status |
|------|-------|-----|--------|
| `w`/`W` | RWKV-7 TimeMix (fla chunk_rwkv7) | CMix / SwiGLU | Default |
| `r`/`R` | RWKV-8 ROSA suffix automaton | CMix / SwiGLU | Available |
| `m`/`M` | Mamba-2 | None / SwiGLU | Optional (`mamba-ssm` extra) |
| `t`/`T` | MHA | None / SwiGLU | **Not functional** — `modules/mha.py` missing |

### RWKV-7 TimeMix (fla kernels) — critical usage
- Training path: `chunk_rwkv7(r, w, k, v, -kk, kk*a)` — **all inputs must be bf16 on CUDA** and the call **must** be wrapped in `torch.amp.autocast("cuda", enabled=False)` to prevent AMP dtype conflicts. Input shapes `[B, T, H, K]`, returns `(output [B, T, H, V], final_state [N, H, K, V])`.
- Inference path: manual recurrence `S[t] = S[t-1] * w + S[t-1] @ ab + vk`.
- Recurrent state: `(B, H, head_size, head_size)` — O(1) in sequence length.
- **Value-first sharing**: layer 0 sets `v_first`, subsequent layers lerp via LoRA gate. `v_first = [None]` is a mutable list per-Isotropic module.
- `seq_len < num_heads` fla warning is benign on short chunked sequences.

### Dynamic chunking
`modules/dc.py`. `RoutingModule` predicts boundaries via cosine similarity between consecutive states. `ChunkLayer` downsamples by keeping boundary tokens. `DeChunkLayer` reconstructs via a custom `_ema_scan` (list accumulation + `torch.stack`, **not** `y[:, t] = ...` inplace — that breaks autograd).

### Agent swarm (scaffold)
`src/aegir/swarm/` — LatentMAS-inspired state fusion, K2.5 PARL orchestration.
- `RWKVStateFusion`: `weighted_sum` / `gated` / `concat_project` modes
- `AlignmentProjection`: cross-agent state space projection
- `FrozenSpecialist`: wraps any Aegir model with `requires_grad_(False)`
- `SwarmOrchestrator`: routing + fusion + PARL reward structure

## Development Environment

Managed by **devenv** (Nix) with **direnv** auto-activation via `.envrc`. Python 3.12 + **uv**; devenv also provides just, cmake, ninja, protobuf, flatbuffers, grpcurl, and mdbook with d2/katex/mermaid plugins.

## Commands

```bash
# IMPORTANT: always pass --no-sync to avoid clobbering patched CUDA extensions
uv run --no-sync python main.py                                             # Shape/forward smoke tests
uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3  # Training smoke test (synthetic data)

# Multi-GPU training (e.g. 6x RTX 4090)
uv run --no-sync torchrun --nproc_per_node=6 train.py --smoke-test --model-size small --epochs 30

# Docs
mdbook build docs/
mdbook serve docs/

# Package management — `uv sync` and bare `uv run` will re-resolve deps and wipe patched wheels
uv add <package>
```

There is no pytest suite; `main.py` and `train.py --smoke-test` are the primary verification path. Checkpoints land in `outputs/best_model.pt`.

### Model sizes (defined in `train.py::make_model`)

| Size | `d_model` | `arch_layout` | Approx params |
|------|-----------|---------------|---------------|
| tiny | `[128, 192, 192]` | `["w2", ["w2", ["w4"], "w2"], "w2"]` | ~13.5M |
| small | `[256, 384, 384]` | `["w4", ["w4", ["w8"], "w4"], "w4"]` | — |
| base | `[768, 1024, 1024]` | `["w4", ["w4", ["w12"], "w4"], "w4"]` | — |

## CUDA Extension Build Notes

flash-attn and (optionally) mamba-ssm/causal-conv1d require patched builds due to CXX11 ABI mismatch between the nix/devenv environment (GCC 15, `_GLIBCXX_USE_CXX11_ABI=1`) and torch's cu124 wheels (`_GLIBCXX_USE_CXX11_ABI=0`).

**To rebuild**: Use `env -i` with system GCC-11, patch setup.py to add explicit `_abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=0"` on both CXX and NVCC args, and set `MAMBA_FORCE_BUILD=TRUE` / `FLASH_ATTENTION_FORCE_BUILD=TRUE` (skips the `CachedWheelsCommand` prebuilt-wheel download). Patched source trees live in `/tmp/mamba_src/`, `/tmp/flash_src/`. See `docs/notes/2026-03-28/010808_deps_smoke_train.md` for the full procedure.

**Prevent clobbering**: Always use `uv run --no-sync`.

**NVIDIA libs**: The devenv venv may need nvidia .so symlinks from the system Python (e.g. libcudnn, libnccl). These are symlinked from `~/.local/lib/python3.10/site-packages/nvidia/*/lib/` to the devenv venv.

Working versions: flash-attn 2.8.3, mamba-ssm 2.3.1, causal-conv1d 1.6.1, Python 3.12, CUDA 12.4.

## Runtime Env Knobs

- `AEGIR_DECHUNK_SCAN` (default `auto`): `auto` picks SSD on CUDA+mamba-ssm, falls back to sequential. Override with `sequential` (debug) or `ssd` (force).
- `AEGIR_DECHUNK_SSD_MIN_L` (default 256): minimum post-chunk sequence length for SSD to be profitable. Below this, sequential is faster because SSD's `chunk_size=64` padding + kernel launch cost dominates.
- `AEGIR_DECHUNK_SSD_HEADDIM` (default 64): SSD internally slices `D` into `nheads` of this size. Needed because single-head-with-large-`headdim` overflows shared memory in the backward kernel on RTX 4090 / Ada when `D ≥ 256`. Must divide `D`.

## Common Pitfalls

- **Inplace ops on autograd tensors**: `y[:, t] = ...` breaks `backward()`. Use a list + `torch.stack`. This already caught the `_ema_scan` once — do not reintroduce.
- **AMP + fla**: wrap `chunk_rwkv7` calls in `torch.amp.autocast("cuda", enabled=False)` and cast inputs to bf16 manually, or you get `"Both operands must be same dtype. Got bf16 and fp32"`.
- **ROSA backward**: `_RosaQKV1BitOp.backward` returns `None` for Q/K/V (non-differentiable suffix automaton) and passes through the `emb` gradient.
- **Empty config lists**: `get_stage_cfg` returns `None` for out-of-range indices (e.g. `window_size=[]`); downstream code must handle this.
- **`group_params`**: must initialize `_optim = {}` on every param before grouping, or `AttributeError`.
- **`t`/`T` blocks**: declared in `block.py` but `modules/mha.py` doesn't exist — use `w`/`W` instead.
- **`mask` vs packed mode**: `AegirForCausalLM.forward` rejects `position_ids` and branches on `mask is None` to switch between unpacked (mask provided) and packed (`cu_seqlens` computed) modes.

## Project Structure

- `main.py` — Forward-pass smoke tests
- `train.py` — DDP/AMP training with cosine LR, load balancing loss, and a synthetic data mode
- `src/aegir/`
  - `models/config.py` — `AegirConfig`, `SSMConfig`, `AttnConfig`, `RWKVConfig` dataclasses
  - `models/aegir.py` — Recursive hierarchical backbone (adapted from H-Net)
  - `models/heads.py` — `AegirForCausalLM` + `AegirForColumnAnnotation` (CTA single-label / CPA multi-label)
  - `modules/rwkv7_tmix.py` — RWKV-7 full TimeMix (fla `chunk_rwkv7`)
  - `modules/rwkv.py` — RWKV-8 ROSA time-mix + relu² CMix + `RWKVBlockState`
  - `modules/rosa.py` — ROSA suffix automaton (CPU, from RWKV-v8)
  - `modules/dc.py` — Dynamic chunking (`RoutingModule`, `ChunkLayer`, `DeChunkLayer` with `_ema_scan`)
  - `modules/isotropic.py` — Flat mixed-block stack + layout-string parser
  - `modules/block.py` — `create_block` factory (w/W/r/R/t/T/m/M), `LayerNormPrenorm` fallback
  - `modules/mlp.py` — SwiGLU
  - `swarm/` — `state_fusion`, `alignment`, `specialist`, `orchestrator`
  - `data/` — `serialization` (table→bytes), `context_select` (MMR), `table_dataset` (CTA/CPA benchmarks)
  - `utils/train.py` — `load_balancing_loss`, `group_params`, `f1_score_multilabel`
- `ref/` — Reference papers (PDFs)
- `docs/` — mdbook site; `docs/notes/YYYY-MM-DD/HHMMSS_*.md` for work summaries (per global `CLAUDE.md` convention)

## Key Reference Codebases (checked out locally)

- H-Net: `~/local/src/wxs/hnet/` — dynamic chunking, `Isotropic` module, hierarchical architecture
- RWKV-LM: `~/local/src/oss/rwkv-lm/` — RWKV-7/8 models, ROSA implementation
- REVEAL: `~/local/src/oss/reveal/` — CTA/CPA benchmarks, MMR context selection, evaluation
