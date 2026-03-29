# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aegir is a hierarchical sequence modeling system using an all-RWKV-7 architecture with H-Net-style dynamic chunking for adaptive segmentation. Primary use case is relational data warehouse metadata tagging (Column Type Annotation and Column Property Annotation for wide tables). Includes agent swarm scaffold (LatentMAS-style RWKV state fusion + K2.5 PARL orchestration). Pre-alpha stage.

Reference papers in `ref/`: H-Net (Dynamic Chunking), Retrieve-and-Verify, ROSA-Tuning, LatentMAS, Kimi K2.5 PARL.

## Architecture

The model uses a recursive hierarchical structure defined by `arch_layout`:
- **All stages**: RWKV-7 TimeMix blocks (`w`/`W`) using flash-linear-attention (fla) Triton kernels
- **Dynamic chunking**: RoutingModule predicts boundaries via cosine similarity; ChunkLayer downsamples; DeChunkLayer reconstructs via EMA scan
- **Optional**: ROSA suffix automaton blocks (`r`/`R`), MHA (`t`/`T`), Mamba-2 (`m`/`M` — requires optional mamba-ssm dep)

Block type codes: `w`=RWKV7+CMix, `W`=RWKV7+SwiGLU, `r`=ROSA+CMix, `R`=ROSA+SwiGLU, `t`=MHA, `T`=MHA+SwiGLU, `m`=Mamba2, `M`=Mamba2+SwiGLU

Example layout: `["w2", ["w2", ["w4"], "w2"], "w2"]` — 2-stage hierarchy with 4 RWKV-7 blocks at core.

### RWKV-7 TimeMix (fla kernels)
- Training: `chunk_rwkv7(r, w, k, v, -kk, kk*a)` — all inputs must be bf16, wrapped in `torch.amp.autocast("cuda", enabled=False)`
- Inference: manual recurrence `S[t] = S[t-1] * w + S[t-1] @ ab + vk`
- Recurrent state: `(B, H, head_size, head_size)` — O(1) in sequence length
- Value-first sharing: layer 0 sets `v_first`, subsequent layers lerp via LoRA gate

### Agent Swarm (scaffold)
- `src/aegir/swarm/` — LatentMAS-inspired state fusion, K2.5 PARL orchestration
- State fusion modes: weighted_sum, gated, concat_project
- FrozenSpecialist wraps any Aegir model with `requires_grad_(False)`

## Development Environment

Environment is managed by **devenv** (Nix-based) with **direnv** integration. Entering the directory auto-activates via `.envrc`. Key tooling provisioned by devenv:

- Python 3.12 with **uv** (package manager, venv, sync)
- just, cmake, ninja, protobuf, flatbuffers, grpcurl
- mdbook (with d2, katex, mermaid plugins) for documentation

## Commands

```bash
# IMPORTANT: Use --no-sync to prevent uv from clobbering patched CUDA extensions
uv run --no-sync python main.py          # Smoke tests
uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3  # Training smoke test

# Multi-GPU training (6x RTX 4090)
uv run --no-sync torchrun --nproc_per_node=6 train.py --smoke-test --model-size small --epochs 30

# Build/serve documentation
mdbook build docs/
mdbook serve docs/

# Package management (caution: uv sync will rebuild CUDA extensions from PyPI)
uv add <package>
uv sync
```

## CUDA Extension Build Notes

flash-attn and (optionally) mamba-ssm/causal-conv1d require patched builds due to CXX11 ABI mismatch between the nix/devenv environment (GCC 15, `_GLIBCXX_USE_CXX11_ABI=1`) and torch's cu124 wheels (`_GLIBCXX_USE_CXX11_ABI=0`).

**To rebuild**: Use `env -i` with system GCC-11, patch setup.py to add explicit `_abi_flag`, and set `MAMBA_FORCE_BUILD=TRUE` / `FLASH_ATTENTION_FORCE_BUILD=TRUE`. Patched source trees: `/tmp/mamba_src/`, `/tmp/flash_src/`. See `docs/notes/2026-03-28/010808_deps_smoke_train.md` for full procedure.

**Prevent clobbering**: Always use `uv run --no-sync` instead of `uv run` to avoid re-resolving deps.

**NVIDIA libs**: The devenv venv may need nvidia .so symlinks from the system Python (e.g. libcudnn, libnccl). These are symlinked from `~/.local/lib/python3.10/site-packages/nvidia/*/lib/` to the devenv venv.

## Project Structure

- `main.py` — Smoke tests for model instantiation and forward pass
- `train.py` — Full training script (DDP, AMP, cosine LR, load balancing loss, synthetic data mode)
- `src/aegir/` — Main package
  - `models/config.py` — AegirConfig, SSMConfig, AttnConfig, RWKVConfig dataclasses
  - `models/aegir.py` — Recursive hierarchical backbone (adapted from H-Net)
  - `models/heads.py` — AegirForCausalLM (pretraining) + AegirForColumnAnnotation (CTA/CPA)
  - `modules/rwkv7_tmix.py` — RWKV-7 full TimeMix (fla chunk_rwkv7 kernels)
  - `modules/rosa.py` — ROSA suffix automaton (CPU-based, from RWKV-v8)
  - `modules/rwkv.py` — RWKV-8 ROSA time mixing + relu² channel mixing + RWKVBlockState
  - `modules/dc.py` — Dynamic chunking (RoutingModule, ChunkLayer, DeChunkLayer with EMA scan)
  - `modules/isotropic.py` — Non-hierarchical sequence processor (mixed block stack)
  - `modules/block.py` — Block factory (create_block with w/W/r/R/t/T/m/M support)
  - `swarm/` — Agent swarm: state_fusion, alignment, specialist, orchestrator
  - `data/` — Table serialization, MMR context selection, benchmark datasets
  - `utils/train.py` — Load balancing loss, parameter grouping, F1 metrics
- `ref/` — Reference papers (PDFs)
- `docs/` — mdbook documentation

## Key Reference Codebases

- H-Net: `~/local/src/wxs/hnet/` — dynamic chunking, Isotropic module, hierarchical architecture
- RWKV-LM: `~/local/src/oss/rwkv-lm/` — RWKV-7/8 models, ROSA implementation, training infra
- REVEAL: `~/local/src/oss/reveal/` — CTA/CPA benchmarks, MMR context selection, evaluation
