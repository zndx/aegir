# Development Guide

## Building and Running

### Critical: Always Use `--no-sync`

```bash
uv run --no-sync python main.py
```

The `--no-sync` flag prevents `uv` from re-resolving and reinstalling dependencies before running. This is **required** because `flash-attn`, `flash-linear-attention` (fla), `mamba-ssm`, and `causal-conv1d` are patched CUDA extensions that were built manually with corrected CXX11 ABI flags. Running `uv run` without `--no-sync` will clobber these patched builds with incompatible PyPI wheels.

### Smoke Tests

```bash
# Model instantiation and forward pass shapes
uv run --no-sync python main.py

# Training loop validation (tiny model, synthetic data)
uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3
```

### Multi-GPU Training

```bash
# 6x RTX 4090 training
uv run --no-sync torchrun --nproc_per_node=6 train.py \
    --model-size small \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4
```

Training uses DDP (DistributedDataParallel), AMP with bf16, cosine LR schedule with linear warmup, and load balancing loss for dynamic chunking regularization.

## CUDA Extension Build Notes

The devenv/Nix environment provides GCC 15, which sets `_GLIBCXX_USE_CXX11_ABI=1`. However, PyTorch's cu124 wheels are built with `_GLIBCXX_USE_CXX11_ABI=0`. This ABI mismatch causes segfaults when CUDA extensions link against the wrong ABI.

### Patching Procedure

Both `mamba-ssm` and `flash-attn` have a `CachedWheelsCommand` in their `setup.py` that downloads prebuilt wheels from GitHub releases, bypassing local compilation. To force a local build with the correct ABI:

1. Set environment variables to force local build:
   ```bash
   export MAMBA_FORCE_BUILD=TRUE
   export FLASH_ATTENTION_FORCE_BUILD=TRUE
   ```

2. Use `env -i` with system GCC-11 to get the correct ABI:
   ```bash
   env -i PATH=/usr/bin:$PATH HOME=$HOME \
       pip install --no-build-isolation /tmp/mamba_src/mamba_ssm-2.3.1/
   ```

3. Patch `setup.py` in each extension to add explicit `_abi_flag` matching torch's ABI.

Patched source trees are kept in `/tmp/mamba_src/` and `/tmp/flash_src/`. See `docs/scratch/2026-03-28/010808_deps_smoke_train.md` for the full step-by-step procedure.

### Verifying the Build

After patching, verify that the extensions load correctly:

```bash
uv run --no-sync python -c "import mamba_ssm; print('mamba-ssm OK')"
uv run --no-sync python -c "import flash_attn; print('flash-attn OK')"
uv run --no-sync python -c "from fla.ops.rwkv7 import chunk_rwkv7; print('fla OK')"
```

## Adding New Block Types

The architecture supports mixed block types (Mamba2, MHA, RWKV-7, RWKV-8 ROSA) within a single model. To add a new block type:

### 1. Implement the Mixer Class

Create a new module that implements three methods:

```python
class MyNewMixer(nn.Module):
    def forward(self, hidden_states, inference_params=None, **kwargs):
        """Full-sequence forward pass. Input: (B, L, D). Output: (B, L, D)."""
        ...

    def step(self, hidden_states, inference_params):
        """Single-token autoregressive step. Input: (B, 1, D). Output: (B, 1, D)."""
        ...

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate KV cache or recurrent state for inference."""
        ...
```

### 2. Register in `create_block()`

Add the new type to `src/aegir/modules/block.py`:

```python
def create_block(arch, d_model, ...):
    if arch in ("x", "X"):  # new block type code
        from my_module import MyNewMixer
        mixer_cls = partial(MyNewMixer, **factory_kwargs, layer_idx=layer_idx)
    ...
```

Convention: lowercase letter = mixer only (no MLP), uppercase = mixer + SwiGLU MLP.

### 3. Add to Isotropic Forward Loop

In `src/aegir/modules/isotropic.py`, add the new block type to:

1. The regex pattern that parses layout strings:
   ```python
   layout_parse = re.findall(r"([mMtTrRwWxX])(\d+)", arch_layout)
   ```

2. The forward loop's block-type dispatch:
   ```python
   elif arch in ("x", "X"):
       layer_mixer_kwargs = {}  # or whatever kwargs your mixer needs
       if hidden_states.dim() == 2:
           hidden_states = hidden_states.unsqueeze(0)
           residual = None if residual is None else residual.unsqueeze(0)
   ```

### 4. Test

```bash
# Verify the new block type instantiates and runs
uv run --no-sync python main.py
```

## Project Structure

```
aegir/
  main.py                          -- Smoke tests
  train.py                         -- Training script (DDP, AMP, cosine LR)
  src/aegir/
    models/
      config.py                    -- AegirConfig, SSMConfig, AttnConfig, RWKVConfig
      aegir.py                     -- Recursive hierarchical backbone
      heads.py                     -- AegirForCausalLM, AegirForColumnAnnotation
    modules/
      block.py                     -- Block factory (create_block)
      isotropic.py                 -- Flat block stack with mixed types
      dc.py                        -- Dynamic chunking (RoutingModule, ChunkLayer, DeChunkLayer)
      rwkv7_tmix.py                -- RWKV-7 full TimeMix (fla kernels)
      rwkv.py                      -- RWKV-8 ROSA time mixing + relu^2 channel mixing
      rosa.py                      -- ROSA suffix automaton (CPU-based)
      mlp.py                       -- SwiGLU MLP
    swarm/
      state_fusion.py              -- RWKVStateFusion (3 modes)
      alignment.py                 -- AlignmentProjection (cross-agent state mapping)
      specialist.py                -- FrozenSpecialist wrapper
      orchestrator.py              -- SwarmOrchestrator (K2.5 PARL)
    data/
      serialization.py             -- Table-to-byte-sequence serialization
      context_select.py            -- MMR context column selection
      table_dataset.py             -- PyTorch dataset for table benchmarks
    utils/
      train.py                     -- Load balancing loss, F1 metrics, param grouping
  docs/                            -- mdbook documentation (this book)
  ref/                             -- Reference papers
```

## Documentation

Build and serve the documentation locally:

```bash
just docs-build           # → docs/current/book/
just docs-serve           # http://localhost:3000

# Equivalent invocations without just:
mdbook build docs/current
mdbook serve docs/current
```

The mdbook layout is the standard ``mdbook init`` shape, rooted at
``docs/current/``: configuration in ``docs/current/book.toml``,
sources under ``docs/current/src/``, theme overrides in
``docs/current/theme/``, generated HTML emitted to
``docs/current/book/``. Per-session scratch notes and archived
chapters live outside this book root, under ``docs/scratch/`` and
``docs/archive/`` respectively, so the build picks up only curated
content.

The book uses mdbook with d2 (architecture diagrams) and a
client-side MathJax 3 shim for math, all provisioned by devenv.
