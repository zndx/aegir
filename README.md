# Ægir

**Hierarchical sequence modeling for relational metadata — H-Net dynamic
chunking over an all-RWKV-7 backbone, with an air-gap-ready leaderboard
gateway for reproducible ablations.**

Ægir forks [H-Net](https://arxiv.org/abs/2507.07955) and replaces every stage's
mixer with RWKV-7 time mixing (via `flash-linear-attention` Triton kernels),
keeping H-Net's content-dependent chunking / dechunking / STE-gated residual
but trading Mamba-2 for RWKV-7 at all levels. Primary target: Column Type
Annotation and Column Property Annotation on wide data-warehouse tables, per
the [Retrieve-and-Verify](https://arxiv.org/abs/2508.17203) paradigm. Includes
an air-gap leaderboard UI as a minimal, self-contained alternative to
W&B/MLflow for this workload.

Pre-alpha; v0.3.0.

## What works today

| Area | Status |
|------|--------|
| **Training** | End-to-end on real `gt-signals-dbpedia` GitTables benchmark (120 DBpedia labels / 814 tables). **v2 mixed-corpus byte-level pretrain** finished 2026-04-27 — 122k steps over 2 GB (FineWeb-Edu + SQaLe + SchemaPile + FinePDFs-lab), stratified held-out eval shows non-degenerate representations and ~2 bpb drops on domain-targeted slices. F1 metrics + per-stage boundary diagnostics logged per epoch. |
| **Run artifacts** | JSON sidecars (`outputs/runs/{run_id}/…`) with pre-rendered static Bokeh plots — readable by the gateway without a tracking daemon. |
| **Leaderboard gateway** | FastAPI on port 8091 serving the React UI + `/api/{health,leaderboard,runs,ontology,classifications}`. All endpoints read-only, no auth, air-gap-friendly. |
| **React UI** | Ant Design shell with Leaderboards / Classifications / Ontologies panels; BokehJS bundled locally, no CDN. |
| **BDD** | 21 tier-0 scenarios + 4 tier-1 training scenarios; `just bdd-0` in ~9 s, `just behave` (full convergence run) in ~3.5 min. |
| **Deployment** | devenv (local), CAI (PGlite + start-app.sh), Zarf package for air-gap K8s (AWS EKS + Cloudera AIIS). |
| **CUDA extensions** | ABI-patched flash-attn 2.8.3 + mamba-ssm 2.3.1 built from source; auto-reinstalled by devenv after every `uv sync`. |

## Quickstart

Prerequisites: [devenv](https://devenv.sh) + [direnv](https://direnv.net), CUDA 12.4 toolkit, gcc-11, an NVIDIA GPU (sm_80+).

```bash
git clone https://github.com/zndx/aegir.git
cd aegir
direnv allow                  # picks up .envrc → devenv shell
just build-flash-attn         # once, ~25 min on a 6×4090 box. Produces
                              # build/wheels/{flash_attn,mamba_ssm}-*.whl.
                              # After this, every ``devenv up`` auto-reinstalls
                              # the patched wheels through the
                              # ``aegir:cuda-ext-reinstall`` task.
devenv up -d                  # postgres:5555, qdrant:6355, gateway:8091, vite:5173
```

Then:

- **Gateway health**: `curl localhost:8091/api/health`
- **UI (dev)**: <http://localhost:5173> — hot-reload React proxying `/api` to the gateway
- **UI (bundle)**: `just ui-build` → gateway serves `ui/dist/` directly from `/`
- **Run the BDD suite**: `just bdd-0` (fast, no GPU) / `just bdd-1` (GPU, no training) / `just behave` (full + `@slow` convergence)
- **Train a real model**: `just run-train --task gt-signals-dbpedia --model-size tiny --epochs 3` → new row appears in the leaderboard on refresh.

## Architecture, briefly

**Core model** — `src/aegir/models/aegir.py`. Recursive hierarchy defined by
`arch_layout` (nested list). At each non-innermost stage: encoder (Isotropic)
→ `RoutingModule` → `ChunkLayer` → main_network (another Ægir) → `DeChunkLayer`
→ decoder, with an STE-gated residual skip around the chunk/main/dechunk block.

**Block alphabet** — `w`/`W` RWKV-7 (default), `r`/`R` ROSA suffix automaton,
`m`/`M` Mamba-2 (optional), `t`/`T` MHA (not functional yet). Lowercase = CMix
relu² FFN; uppercase = SwiGLU. See `CLAUDE.md` for traps (AMP-fla dtype contract,
inplace-autograd in EMA scan, ROSA `step()` raises `NotImplementedError`
rather than silently decoding zeros, etc.).

**Runtime knobs** (env vars):

- `AEGIR_DECHUNK_SCAN` (default `auto`) — EMA scan backend. `auto` uses the
  mamba-ssm SSD kernel on CUDA (~2× wall-clock speedup vs sequential at
  `L=1024` on real workloads); `sequential` forces the reference
  implementation; `ssd` forces the kernel (will fail cleanly on CPU).
- `AEGIR_DECHUNK_SSD_MIN_L` (default 256) — short-L threshold below which
  the dispatcher falls back to sequential (SSD's kernel-launch overhead
  outweighs savings for tiny post-chunk sequences).
- `AEGIR_DECHUNK_SSD_HEADDIM` (default 64) — head-slicing granularity.
  Needed because single-head-with-large-`headdim` overflows RTX 4090 /
  Ada shared memory in the backward CB kernel when `D ≥ 256`.
- `AEGIR_MMR_CACHE_DIR` / `AEGIR_MMR_CACHE_DISABLE` — disk-backed
  SQLite+blake2b cache for MPNet embeddings used by MMR context selection.
  Default on; 20,000× cache-hit speedup over cold encode.

**Leaderboard stack** — Python: `src/aegir/gateway/app.py` (FastAPI, ~220 LOC).
UI: `ui/src/` (React 19 + Ant 5 + Vite 6, stripped from Atelier's shell).
Data path: `train.py` → `RunArtifacts` (`src/aegir/utils/runs.py`) writes
`metadata.json` + `metrics.json` + 4 static Bokeh plots (loss / F1 / per-stage
boundary). Gateway scans `outputs/runs/` and serves the Bokeh JSON via
`/api/runs/{id}/plot/{name}` for `Bokeh.embed.embed_item` in the browser.
No Panel server, no DynamicMap, no external tracking daemon.

## BDD tiers

Mirrors [Atelier's](../atelier/) tagging tactic. Scenarios carry observable
assertions (F1 > threshold, boundary selection rate within tolerance) rather
than ceremony-heavy unit-style checks.

```
just bdd-0          # 21 scenarios, no GPU, no fixtures, ~9 s.  Pre-commit gate.
just bdd-1          # adds the 2 @tier-1 scenarios that need GPU + fixture.
just behave         # full + @slow (real training, 3 epochs on gt-signals-dbpedia).
just behave-one features/chunking/dynamic_boundaries.feature
```

Env var gates: `AEGIR_BDD_TIER={0,1}` and `AEGIR_BDD_SLOW={0,1}`, set by the
Justfile recipes.

## Deployment

| Target | Datastore | Gateway | Launch |
|--------|-----------|---------|--------|
| **devenv** (dev) | Postgres 16 @ 5555 (devenv service, pgvector enabled) | 8091 | `devenv up -d` |
| **CAI** | PGlite @ 5545 (WASM, bundled via `scripts/pglite-server.mjs`) | `$CDSW_APP_PORT` | `bin/start-app.sh` |
| **Zarf air-gap K8s** | `pgvector/pgvector:pg16` StatefulSet | `Deployment` + `Ingress` @ `aegir.${domain}` | `zarf package deploy …` — supports `TARGET=cloudera-aiis` overlay for Cloudera AI Inference Service |

Port allocation deliberately non-conflicting with sibling Cloudera projects on
the same dev box (Atelier 5533/6333, Gaius 5444/6339, Cybersec 5438, Signals
5455). See `config/base.conf` for the HOCON schema with `${?ENV}` overrides.

Build the Zarf package:

```bash
just ui-build
just zarf-build   # → zarf-package-aegir-leaderboard-amd64-0.1.0.tar.zst
```

Full runbook at `zarf/README.md`.

## CUDA extensions — why the auto-reinstall task exists

torch 2.6+cu124 is compiled with `_GLIBCXX_USE_CXX11_ABI=0`. Both flash-attn's
and mamba-ssm's GitHub prebuilt wheels (`cxx11abiTRUE` **and** `cxx11abiFALSE`
variants) link against the `__cxx11::basic_string` form of `c10::Error::Error`,
which torch's libc10 defines only in the old-ABI form. Result: `undefined
symbol` on import. The recipe:

- `just build-flash-attn` — source build in `build/patched-src/`, single-arch
  `sm_89`, `NVCC_THREADS=1`, `MAX_JOBS=16`, 16 GB safety-net swapfile, 15-min
  stall monitor, cleanup trap. ~25 min wall time on 6×4090. Produces
  `build/wheels/*.whl`.
- `devenv.nix` → `tasks."aegir:cuda-ext-reinstall"` runs after
  `devenv:python:uv` and before `devenv:enterShell`. Reinstalls from
  `build/wheels/` every `devenv up` / `devenv shell`, so `uv sync` clobbering
  the patched install is harmless. Emits `[aegir] restoring patched CUDA
  extensions: …` on the way in.

No patched wheels under `build/wheels/`? The task emits a hint and continues —
`flash_attn` / `mamba_ssm` imports fall back to `LayerNormPrenorm` / the
non-Mamba path until you run `just build-flash-attn`. Nothing in M1 hard-depends
on either extension.

## Status & roadmap

- **M0** — end-to-end BDD-backed training on real `gt-signals-dbpedia`,
  boundary diagnostics visible per epoch. ✅
- **M1** — leaderboard gateway + UI, air-gap deployment envelope (devenv /
  CAI / Zarf), ABI-patched CUDA extensions. ✅
- **M2** — *in progress.*  v2 → SOTAB head fine-tune with liveness gate
  (≥ 0.10 macro F1, ≥ 3 MCL clusters, ≥ 10 distinct predicted labels);
  ontology + synth ownership migration (`src/aegir/ontology/` and
  `src/aegir/synth/` greenfield); `_LABEL_DIMS["sotab"] = 91 → 82`
  reconciliation; `vocab_label_map.json` v1.0.0 as the first outward
  contract; external-baseline harness (Nemotron 3 Nano, OpenAI OSS 20b,
  REVEAL reimpl); per-class F1 bars in the leaderboard UI.
- **M3** — vocabulary expansion past the migrated baseline (Ægir-defined,
  not externally-tiered); multi-GPU step-up (8 GB on 6×4090 ≈ 7 h);
  v3 corpus mix; KServe `InferenceService` predictor for online inference
  on Cloudera AI Inference Service; GPU-flavored Zarf image bundling a
  trained checkpoint; Datashader/Dask for large-run visualization.
- **M4** — competitive F1 against published baselines on SOTAB-CTA,
  GitTables, WikiTables; `base` config (~500M params).

Full roadmap with empirical gates and the far-future K2.5 RL post-training
track is in `docs/current/src/roadmap.md`. The ontology contract and migration
plan are in `docs/current/src/ontology/`.

## References

- H-Net: [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- REVEAL: [Retrieve-and-Verify Table Context Selection](https://arxiv.org/abs/2508.17203)
- ROSA: [Enhancing Long-Context Modeling via Suffix Matching](https://arxiv.org/abs/2502.XXXXX)
- RWKV-7 kernels: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- LatentMAS (inspiration, not direct port): [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)
- Kimi K2.5 PARL (inspiration): [Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
- Project docs: `docs/current/` (mdbook root; build via `just docs-build`).
  Architecture details: `CLAUDE.md` for the traps + block-type truth table.
