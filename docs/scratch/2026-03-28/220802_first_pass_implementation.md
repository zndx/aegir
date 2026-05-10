# First-Pass Implementation of Aegir Model Architecture

## Summary

Implemented the complete `src/aegir/` package structure as a first-pass scaffolding of the Aegir model — a novel hierarchical sequence model combining RWKV-8 (with ROSA suffix automaton) at the innermost stage, H-Net-style dynamic chunking between stages, and Mamba-2 encoder/decoder at outer stages.

## What was implemented

### Core modules (ported/adapted from reference codebases)

| Module | Source | Description |
|--------|--------|-------------|
| `modules/rosa.py` | RWKV-v8 | ROSA suffix automaton (QKV matching, 1-bit layer) |
| `modules/rwkv.py` | RWKV-v8 | RWKV-8 blocks (ROSA time mixing + relu² channel mixing) |
| `modules/dc.py` | H-Net | Dynamic chunking (RoutingModule, ChunkLayer, DeChunkLayer) |
| `modules/isotropic.py` | H-Net | Non-hierarchical sequence processor with mixed block types |
| `modules/block.py` | H-Net (extended) | Block factory supporting m/M/t/T/r/R architecture codes |
| `modules/mlp.py` | H-Net | SwiGLU feedforward network |
| `modules/utils.py` | H-Net | Helpers (get_seq_idx, get_stage_cfg, apply_optimization_params) |

### Model architecture

| Module | Description |
|--------|-------------|
| `models/config.py` | AegirConfig, SSMConfig, AttnConfig, RWKVConfig dataclasses |
| `models/aegir.py` | Recursive hierarchical backbone (Aegir class) |
| `models/heads.py` | AegirForCausalLM (pretraining) + AegirForColumnAnnotation (CTA/CPA) |

### Data pipeline

| Module | Source | Description |
|--------|--------|-------------|
| `data/context_select.py` | REVEAL | MMR context column selection |
| `data/serialization.py` | REVEAL | Table → token sequence with role markers |
| `data/table_dataset.py` | REVEAL | 6 benchmark dataset class stubs |

### Training utilities

| Module | Source | Description |
|--------|--------|-------------|
| `utils/train.py` | H-Net + REVEAL | Load balancing loss, param grouping, F1 metrics |

## Key design decision

RWKV-8 integration happens through the block factory (`'r'` code) within the existing Isotropic framework, rather than as a separate module class. This means the recursive Aegir hierarchy is structurally identical to H-Net — the arch_layout string `["m4", ["m4", ["r12"], "m4"], "m4"]` naturally places RWKV blocks at the innermost stage.

## What's next

1. Install dependencies and verify smoke test (`uv run python main.py`)
2. Implement concrete `_load_tables()` methods for each benchmark dataset
3. Add training script with PyTorch Lightning or plain training loop
4. Add CUDA kernels for ROSA (currently CPU-only, slow)
5. Pretraining on language modeling data before CTA/CPA finetuning
