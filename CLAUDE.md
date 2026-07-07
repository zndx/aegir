# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aegir is a hierarchical sequence modeling system using an all-RWKV-7 architecture with H-Net-style dynamic chunking for adaptive segmentation. Primary use case is relational data warehouse metadata tagging (Column Type Annotation and Column Property Annotation for wide tables). Includes agent swarm scaffold (LatentMAS-style RWKV state fusion + K2.5 PARL orchestration). Pre-alpha stage.

The project has two parallel tracks:
- **The model** (v0.2, `src/aegir/models|modules|swarm/`): the RWKV-7 + dynamic-chunking backbone, documented under [Architecture](#architecture).
- **The ontology-grounded synthetic-corpus pipeline** (v0.3, `src/aegir/ontology/` + `scripts/`): the current focus of active work. A 540-template OWL/BFO ontology drives LLM generation of verifiable, attribution-clean textbook chapters used as byte-level pretraining data. Documented under [Ontology & Synthetic Corpus Pipeline](#ontology--synthetic-corpus-pipeline-v03). See `docs/scratch/2026-05-18/004500_v0_3_plan_of_record.md` for the 9-artifact release plan.

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

## Ontology & Synthetic Corpus Pipeline (v0.3)

The v0.3 thesis: a byte-level model trained on deterministically-verifiable, ontology-grounded synthetic textbook chapters can match TAPAS/TAPEX-class column-annotation performance **while** producing a corpus that is itself publishable and attribution-clean. The pipeline is a closed loop: ontology → coverage audit → chapter generation → verification → (optionally) byte-pretraining corpus.

### Ontology (`src/aegir/ontology/`)
A content-first DERIVED template catalog grounded in BFO 2020 / CCO (the hand-authored 01-07 seed families were retired in 8df0d23 — everything is derived now). Each template is a Manchester-syntax axiom skeleton with typed slots (`{name:Type}` / `{name:Type:Bound}` — see `SLOT_DSL.md`) carrying provenance (`pattern`/`tier`/`grounds_ddl`/`domain`/`source_span`).

- `schema.py` — `CatalogTemplate` / `Catalog` dataclasses + `load_catalog`/`save_catalog` + `catalog_files()` (the canonical discovery helper — use it, not globs). Templates carry `manchester_template`, `slot_types`, `is_complex`, `verbal_template` (DeepOnto-derived), `bfo_anchor_path`, and `provenance`.
- `catalog/` — `catalog.json` (THE live catalog; formerly `08_derived.json`), `catalog.candidate.json` (deriver staging — promoted by `promote_candidates.py`), `combined.json` (regenerated), and `null_stats` for R_D normalization. **Edit `catalog.json` (or better: let the derive→promote loop accrete it), never `combined.json`.**
- `verifier.py` — the runtime efficacy score **R = R_A·(0.50·R_B + 0.05·R_C + 0.45·R_D)** (weights locked from a P2 sweep; `tune_verifier_weights.py` re-derives them). `R_A` is a hard structural type-gate (slots present + type-match), `R_B` complexity saturation, `R_C` semantic richness, `R_D` topic alignment to corpus.
- `topic_alignment.py` — `R_D`: encode (all-MiniLM-L6-v2) → KMeans → Hungarian-matched cosine between generated-text centroids and the cached corpus topic model (`T_I.pkl`, fit by `build_topic_model.py`). Encoder is cached in-process.
- `complex.py` — **family simplicial complex**: the family-set a chapter cites is a simplex; `family_complex.json` records empirically-measured `maximal_simplices` (R_axiom ≥ floor, n ≥ min_count) and `measured_below_floor` "punctures" (simplices observed to fail — these do **not** propagate to supersets). `FamilyComplex.is_allowed(simplex)` gates generation; `best_face(candidate)` returns the largest allowed subset so the sampler can drop offending templates. Built by `build_family_complex.py` from chapter runs.
- `deeponto_harness.py` — **offline-only** JVM harness (DeepOnto verbalizer) run at catalog-build time to populate `verbal_template`/`is_complex`/`mean_verbal_length`. Gotchas: DeepOnto calls `click.prompt()` at import (hangs non-interactively until the JVM starts), and system openjdk-11 needs a bootstrapped `libz.so.1` because Nix masks `/usr/lib` — hence the `LD_LIBRARY_PATH=$(pwd)/build/jvm-libs` prefix on ontology commands.

### Chapter generation & verification (`scripts/`)
- `generate_chapter.py` — ontology-grounded chapter synthesis. **Topic-first sampler**: pick a target FinePDFs topic + style siblings with anti-repetition weighting (∝ 1/(1+usage)), select K templates (family-diverse round-robin by default), enforce the family complex via `best_face`, verbalize axioms + attach style-anchor passages, then call a weighted **GLM-4.7 / Grok-4.3 mix** (`--mix cerebras/zai-glm-4.7:0.6,xai/grok-4.3:0.4`). Three ablation arms via `--ablation {full,no-ontology,no-schema}` (each has its own prompt template). Full exchange (incl. `reasoning_content`) is captured to an Iceberg `raw.exchange` table; chapters land in `chapters.parquet` + per-chapter `.md`. Requires `--audit-run <coverage_v0/run_id>/`.
- `verify_chapters.py` — **4-scorer verification loop**: `R_topic` (sentence-transformer cosine to style anchors), `R_iri` (cited-template keyword presence in prose), `R_density` (markdown-table structure: ≥2 tables GLM / ≥3 with cross-FKs Grok), `R_axiom` (table headers match slot types). Composite = geometric mean; status = accepted (≥`--tau-accept` 0.50) / borderline / rejected (<`--tau-review` 0.30). `--write-to-iceberg` appends to `raw.chapter_verification`.
- `ontology_coverage_audit.py` — produces the `coverage_v0/<run_id>/` inputs both scripts consume: `topic_coverage.parquet`, `template_density.parquet`, `family_density.parquet` (KMeans-cluster FinePDFs, score cosine similarity of each topic to all 540 templates).
- `scripts/experiment_*.py` — standalone exploratory probes (BERTopic at various scales, loose-Grok scaffolding tests, topic-recovery-rate measurement, SDG verbalization). Not production; safe to read for intent, not wired into the pipeline.

Key finding (per the plan-of-record): the ontology is **load-bearing for slot-type prediction (CPA)** but **trades raw FinePDFs distribution alignment** — so the v0.2 SOTAB-CTA target may be the wrong eval. Don't assume SOTAB-CTA is the goal without checking the current plan.

## Development Environment

Managed by **devenv** (Nix) with **direnv** auto-activation via `.envrc`. Python 3.12 + **uv**; devenv also provides just, cmake, ninja, protobuf, flatbuffers, grpcurl, and mdbook with d2/katex/mermaid plugins.

## Commands

```bash
# IMPORTANT: always pass --no-sync to avoid clobbering patched CUDA extensions
uv run --no-sync python main.py                                             # Shape/forward smoke tests
uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3  # Training smoke test (synthetic data)

# Multi-GPU training (e.g. 6x RTX 4090)
uv run --no-sync torchrun --nproc_per_node=6 train.py --smoke-test --model-size small --epochs 30

# Byte-level pretraining on the synthetic corpus (Phase 0 / 0.5; rank-sharded parquet)
uv run --no-sync python train_pretrain.py --resume-from <ckpt> --phase05-... # see module docstring

# v0.3 ontology pipeline — note the LD_LIBRARY_PATH prefix (DeepOnto/JVM, see ontology section)
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/ontology_coverage_audit.py --output-dir <coverage_v0/>
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/generate_chapter.py --audit-run <coverage_v0/run/> --n-chapters 100 --output <chapters_v0/>
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/verify_chapters.py --chapters-run <chapters_v0/run/> --audit-run <coverage_v0/run/>
just check-ontology-schema   # mechanical CI: TTL parses, labels/definitions present, BFO ancestry, SPARQL totality

# Docs (mdbook layout: book root at docs/current/, sources at docs/current/src/)
just docs-build           # → docs/current/book/
just docs-serve           # → http://localhost:3000

# Package management — `uv sync` and bare `uv run` will re-resolve deps and wipe patched wheels
uv add <package>
```

There is no pytest suite; `main.py` and `train.py --smoke-test` are the primary verification path for the model, and `just check-ontology-schema` for the ontology. Checkpoints land in `outputs/best_model.pt`.

### Model sizes (defined in `train.py::make_model`)

| Size | `d_model` | `arch_layout` | Approx params |
|------|-----------|---------------|---------------|
| tiny | `[128, 192, 192]` | `["w2", ["w2", ["w4"], "w2"], "w2"]` | ~13.5M |
| small | `[256, 384, 384]` | `["w4", ["w4", ["w8"], "w4"], "w4"]` | — |
| base | `[768, 1024, 1024]` | `["w4", ["w4", ["w12"], "w4"], "w4"]` | — |

## CUDA Extension Build Notes

flash-attn and (optionally) mamba-ssm/causal-conv1d require patched builds due to CXX11 ABI mismatch between the nix/devenv environment (GCC 15, `_GLIBCXX_USE_CXX11_ABI=1`) and torch's cu124 wheels (`_GLIBCXX_USE_CXX11_ABI=0`).

**To rebuild**: Use `env -i` with system GCC-11, patch setup.py to add explicit `_abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=0"` on both CXX and NVCC args, and set `MAMBA_FORCE_BUILD=TRUE` / `FLASH_ATTENTION_FORCE_BUILD=TRUE` (skips the `CachedWheelsCommand` prebuilt-wheel download). Patched source trees live in `/tmp/mamba_src/`, `/tmp/flash_src/`. See `docs/scratch/2026-03-28/010808_deps_smoke_train.md` for the full procedure.

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
- `train.py` — DDP/AMP training (CTA/CPA heads) with cosine LR, load balancing loss, and a synthetic data mode
- `train_pretrain.py` — Byte-level Phase 0 / 0.5 pretraining on the synthetic/TAPAS-style corpus; rank-sharded parquet (avoids OOM cascade), `--label-token-weight`, label-aware eval metrics, `--resume-from` checkpoint chaining
- `scripts/` — the v0.3 ontology pipeline (see [Ontology section](#ontology--synthetic-corpus-pipeline-v03)): `ontology_coverage_audit.py`, `generate_chapter.py`, `verify_chapters.py`, `build_catalog.py`, `build_family_complex.py`, `build_topic_model.py`, `tune_verifier_weights.py`, `check_ontology_schema.py`, plus corpus download/eval/`experiment_*` probes
- `src/aegir/`
  - `ontology/` — `schema`, `verifier`, `topic_alignment`, `complex` (family simplicial complex), `deeponto_harness` (offline JVM); `catalog/*.json` (540 templates, 7 families), `family_complex.json`, `T_I.pkl`, `null_stats.json`, `sdg-vocab.ttl`, `SLOT_DSL.md`
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
- `docs/` — mdbook site; `docs/scratch/YYYY-MM-DD/HHMMSS_*.md` for work summaries (per global `CLAUDE.md` convention)

## Key Reference Codebases (checked out locally)

- H-Net: `~/local/src/wxs/hnet/` — dynamic chunking, `Isotropic` module, hierarchical architecture
- RWKV-LM: `~/local/src/oss/rwkv-lm/` — RWKV-7/8 models, ROSA implementation
- REVEAL: `~/local/src/oss/reveal/` — CTA/CPA benchmarks, MMR context selection, evaluation
