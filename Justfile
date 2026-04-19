# Aegir command aliases. Tier semantics mirror ~/local/src/zndx/atelier/justfile.
#
#   just bdd-0     — tier-0, fast, no GPU, no fixtures. Runs on every commit.
#   just bdd-1     — tier-1, GPU + tiny fixture, excludes @slow. ~60 s budget.
#   just behave    — end-to-end: tier-1 + @slow including full training runs.
#                    The "does Aegir actually learn" verdict.
#   just behave-one FEATURE=<path> — run a single feature file at tier-1 + slow.
#
# Ancillaries:
#   just sotab-fixture — rebuild features/fixtures/gittables_tiny/ from cldr/signals drop.

set positional-arguments

default:
    @just --list

bdd-0:
    AEGIR_BDD_TIER=0 uv run --no-sync behave features/

bdd-1:
    AEGIR_BDD_TIER=1 uv run --no-sync behave features/ --tags="~@slow"

behave:
    AEGIR_BDD_TIER=1 AEGIR_BDD_SLOW=1 uv run --no-sync behave features/

behave-one feature:
    AEGIR_BDD_TIER=1 AEGIR_BDD_SLOW=1 uv run --no-sync behave {{feature}}

sotab-fixture:
    uv run --no-sync python scripts/make_tiny_fixture.py

# ── DED (cross-table Data Element Discovery) ──────────────────
#
#   just ded-smoke   — synthetic-data scaffolding proof for AegirForDED +
#                      SupCon. Trains a tiny model on a toy multi-table
#                      cluster task with known ground truth. Expected:
#                      B-cubed F1 rises from ~0.18 (random) to ~0.45 in
#                      3 epochs — a scaffolding sanity-check, not a
#                      benchmark (the synthetic task is easy by design).

ded-smoke *args:
    uv run --no-sync python scripts/train_ded_smoke.py {{args}}

# ── Official benchmark datasets ───────────────────────────────
#
# Both downloaders land under /raid/datasets/ (trivial footprint:
# SOTAB ~4.7 GB, GitTables ~12 GB). Each is idempotent + resumable;
# re-running is a no-op once files verify.
#
#   just get-sotab         — SOTAB V2 (Schema.org + DBpedia, CTA + CPA)
#   just get-gittables     — GitTables 1M (Zenodo record 6517052)
#   just get-benchmarks    — both, sequentially

get-sotab:
    uv run --no-sync python scripts/download_sotab.py

get-gittables:
    uv run --no-sync python scripts/download_gittables.py

get-gittables-quick *args:
    # Fetch only the first N zips — for smoke tests / limited-bandwidth setups.
    uv run --no-sync python scripts/download_gittables.py --limit 5 {{args}}

get-benchmarks:
    #!/usr/bin/env bash
    set -euo pipefail
    just get-sotab
    just get-gittables
    echo "=== All benchmark data in place ==="
    ls -la /raid/datasets/sotab /raid/datasets/gittables | head -30

# ── Model benchmarking ────────────────────────────────────────
#
# ``just benchmarks`` evaluates the current model against every
# registered official benchmark (SOTAB-CTA/CPA Schema.org + DBpedia,
# GitTables-1M DBpedia + Schema.org, plus the Atelier gt-signals
# sample). Each (task, model_size, seed) triple produces one run
# under outputs/runs/ and therefore one leaderboard row.
#
# Mode depends on target:
#   just benchmarks-quick  — tiny model, 1 epoch, small subset per task
#                            (verification; ~5-10 min total)
#   just benchmarks        — tiny model, 3 epochs, full train data
#                            (~30-60 min total on 6×4090)
#   just benchmarks-full   — small model, 30 epochs per task
#                            (hours; real convergence numbers)
#
# The leaderboard UI (http://localhost:8091) picks up new rows on
# next page-refresh — no separate import step.

# ── max_length sizing rationale ────────────────────────────────
#
# Aegir operates on bytes; REVEAL operates on RoBERTa BPE tokens.
# REVEAL's serializer (~/local/src/oss/reveal/src/dataset.py) gives each
# column a per-cell budget of ``512 // num_cols`` *tokens*. For a typical
# SOTAB CTA table of ~9 columns that's ~56 tokens/column ≈ 200 bytes/column
# at BPE's 3-4 char/token average. Total table budget ≈ 500 tokens × 4 =
# 2 kB of bytes.
#
# Aegir max_length maps 1:1 to bytes. Our `serialize_table` divides the
# budget evenly across columns (adaptive_length=True). To match REVEAL's
# per-column context we need:
#   max_length=2048  → ~225 B per column with 9 cols (target+8 context)
#                      ≈ REVEAL's 56 tokens/col — parity
#   max_length=1024  → ~112 B per column — half parity (fewer cells/col)
#   max_length=512   → ~56  B per column — single-digit cells/col, lossy
#   max_length=256   → smoke/verify only
#
# Sizing is the single biggest knob for F1. Start-of-M2 sweep plan:
#   benchmarks-quick   max_length=256, no MMR    ← sanity-check only
#   benchmarks         max_length=1024, MMR=8    ← Phase-1 calibration
#   benchmarks-full    max_length=2048, MMR=8    ← REVEAL-parity attempt
#                                                  (byte-level from scratch;
#                                                  expected to land below
#                                                  REVEAL's 0.815 until we
#                                                  add pretraining per
#                                                  docs/src/pretraining.md)

benchmarks-quick:
    #!/usr/bin/env bash
    # Every registered benchmark, tiny model, 200 train steps + small
    # val subset. Proves the pipeline end-to-end in roughly 2-4 min per
    # task (~20 min total for all 7). Numbers will be near-random;
    # these rows exist to sanity-check loaders, not to report F1.
    set -euo pipefail
    for task in gt-signals-dbpedia sotab sotab-re sotab-dbp sotab-dbp-re gittables-dbpedia gittables-schemaorg; do
        echo
        echo "════════════════════════════════════════════════════════════"
        echo "  benchmark-quick: $task  (tiny, 200 steps, max_length=256, val≤500)"
        echo "════════════════════════════════════════════════════════════"
        uv run --no-sync python train.py --task "$task" \
            --model-size tiny --epochs 1 --batch-size 16 --max-length 256 \
            --max-context-cols 0 \
            --max-train-steps 200 --max-val-samples 500 \
            || echo "[$task] failed — continuing"
    done
    echo
    echo "=== benchmarks-quick complete — see http://localhost:8091/leaderboards ==="

benchmarks:
    #!/usr/bin/env bash
    # Phase-1 calibration run: small model, 3 full epochs, max_length=1024
    # with 8-column MMR context. Half-parity with REVEAL's per-column
    # budget but starting from random weights. Target: docs/roadmap
    # Phase-1 exit criteria (macro F1 > 0.65 hard / > 0.85 easy on
    # SOTAB-CTA), not REVEAL's 0.815 micro.
    #
    # Expected: ~2-3 h per SOTAB task on 6×4090, less for the smaller
    # variants; full batch ~8-14 h.
    set -euo pipefail
    for task in gt-signals-dbpedia sotab sotab-re sotab-dbp sotab-dbp-re gittables-dbpedia gittables-schemaorg; do
        echo
        echo "════════════════════════════════════════════════════════════"
        echo "  benchmark: $task  (small, 3 epochs, max_length=1024, MMR=8)"
        echo "════════════════════════════════════════════════════════════"
        uv run --no-sync python train.py --task "$task" \
            --model-size small --epochs 3 --batch-size 16 --max-length 1024 \
            --max-context-cols 8 --lr 3e-4 \
            || echo "[$task] failed — continuing"
    done
    echo
    echo "=== benchmarks complete — see http://localhost:8091/leaderboards ==="

benchmarks-full:
    #!/usr/bin/env bash
    # Full convergence attempt: base model, 30 epochs, max_length=2048
    # with 8-column MMR context. Byte-level parity with REVEAL's token
    # budget. Target: competitive with REVEAL on CTA/CPA tasks.
    # Expected days-to-weeks of 6×4090 compute depending on task mix.
    set -euo pipefail
    for task in sotab sotab-re sotab-dbp sotab-dbp-re gittables-dbpedia gittables-schemaorg; do
        echo
        echo "════════════════════════════════════════════════════════════"
        echo "  benchmark-full: $task  (base, 30 epochs, max_length=2048, MMR=8)"
        echo "════════════════════════════════════════════════════════════"
        uv run --no-sync python train.py --task "$task" \
            --model-size base --epochs 30 --batch-size 8 --max-length 2048 \
            --max-context-cols 8 --lr 1e-4 \
            || echo "[$task] failed — continuing"
    done

# ── REVEAL-parity recipe ──────────────────────────────────────
#
# Matches REVEAL's hyperparameters as closely as byte-level Aegir can:
# same 8-MMR-context serialization, same ~2kB per-column budget,
# 50 epochs to match their fine-tuning runs. Honest-comparison row for
# the leaderboard; expected numbers below 0.815 until we layer in
# ontology-grounded pretraining (docs/src/pretraining.md).
#
# Override: ``REVEAL_TASKS="sotab sotab-re" just benchmarks-reveal-match``
benchmarks-reveal-match:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${REVEAL_TASKS:=sotab sotab-re sotab-dbp sotab-dbp-re}"
    for task in $REVEAL_TASKS; do
        echo
        echo "════════════════════════════════════════════════════════════"
        echo "  reveal-match: $task  (small, 50 epochs, max_length=2048, MMR=8)"
        echo "════════════════════════════════════════════════════════════"
        uv run --no-sync python train.py --task "$task" \
            --model-size small --epochs 50 --batch-size 16 --max-length 2048 \
            --max-context-cols 8 --lr 3e-4 --warmup-ratio 0.1 \
            || echo "[$task] failed — continuing"
    done
    echo
    echo "=== reveal-match complete — compare rows on leaderboard to REVEAL baselines ==="

# Legacy smoke tests from the pre-BDD era. Still useful for quick sanity.
smoke:
    uv run --no-sync python main.py

train-smoke:
    uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3

# ── M1 deployment envelope ────────────────────────────────────────
#
#   just resolve-config   — materialize build/config/aegir.{env,json} from HOCON
#   just gateway          — run FastAPI gateway (no UI proxy; raw mode)
#   just ui-dev           — Vite dev server on :5173, proxies /api to gateway
#   just ui-build         — produce ui/dist/ for production
#   just run-train ARGS   — real-data training that writes sidecar to outputs/runs/
#   just zarf-build       — build aegir-gateway image + Zarf .tar.zst package
#   just pglite-smoke     — launch PGlite + gateway on the CAI-like port
#
# devenv up brings up postgres+qdrant+gateway+vite-dev as a unit — use the
# targets below to run pieces individually.

resolve-config:
    uv run --no-sync python bin/resolve-config.py

gateway:
    uv run --no-sync python -m aegir.gateway

ui-dev:
    cd ui && pnpm install --silent --prefer-offline && pnpm dev

ui-build:
    cd ui && pnpm install --silent --prefer-offline && pnpm build

run-train *args:
    uv run --no-sync python train.py {{args}}

zarf-build:
    #!/usr/bin/env bash
    set -euo pipefail
    just ui-build
    podman build -t localhost:5555/aegir-gateway:0.1.0 \
        -f zarf/images/Dockerfile.aegir-gateway .
    # Start a podman socket so Zarf can talk to the daemon.
    if ! pgrep -f "podman system service" >/dev/null; then
        podman system service --time=600 unix:///run/user/$(id -u)/podman/podman.sock &
        sleep 1
    fi
    cd zarf
    DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock \
        zarf package create . --confirm --skip-sbom

pglite-smoke:
    AEGIR_GATEWAY_PORT=8100 CDSW_APP_PORT=8100 bash bin/start-app.sh

# ── Patched CUDA extensions ────────────────────────────────────
#
# flash-attn / mamba-ssm / causal-conv1d publish prebuilt wheels on
# GitHub releases, but both `cxx11abiTRUE` and `cxx11abiFALSE`
# variants link against the __cxx11 ABI form of c10::Error::Error
# while torch cu124's libc10 defines only the OLD ABI form
# (``_GLIBCXX_USE_CXX11_ABI=0``). Neither prebuilt wheel is ABI-
# compatible with our torch. See CLAUDE.md for background.
#
# The recipe:
#   - causal-conv1d installs fine from PyPI (prebuilt wheel is
#     built with old ABI, matches torch). Listed for completeness.
#   - mamba-ssm + flash-attn must be built from source with
#     ``_GLIBCXX_USE_CXX11_ABI=0`` (set automatically by setup.py
#     reading torch._C._GLIBCXX_USE_CXX11_ABI), single-arch
#     sm_89 (TORCH_CUDA_ARCH_LIST=8.9), NVCC_THREADS=1 to avoid an
#     nvcc 12.4 segfault in PTX generation on template-heavy .cu
#     files, and ninja on PATH.
#
# ``just cuda-deps`` rebuilds all three from source under
# ``build/patched-src/`` into wheels under ``build/wheels/``.  No
# explicit install step: devenv's ``aegir:cuda-ext-reinstall`` task
# (in devenv.nix) reinstalls from build/wheels/ on every ``devenv up``
# / ``devenv shell``, right after uv sync.  Idempotent.  Target the
# 6×RTX-4090 box only (sm_89 is hardcoded for now).

cuda-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p build/wheels
    # causal-conv1d: prebuilt wheel from PyPI is ABI-compatible.
    # Not built from source; listed here so devenv won't strip it.
    uv pip install --reinstall --no-build-isolation causal-conv1d==1.6.1
    # mamba-ssm: build from source
    rm -rf build/patched-src/mamba_ssm-2.3.1/{build,dist,*.egg-info}
    bash bin/build-cuda-ext.sh build/patched-src/mamba_ssm-2.3.1 MAMBA_FORCE_BUILD
    cp build/patched-src/mamba_ssm-2.3.1/dist/mamba_ssm-*.whl build/wheels/
    # flash-attn: build from source (~25 min on 6×4090 via the
    # aggressive recipe; this fall-back path uses the conservative
    # defaults from build-cuda-ext.sh and takes 2-3 h).
    # For the 25-min path use ``just build-flash-attn`` instead.
    rm -rf build/patched-src/flash_attn-2.8.3/{build,dist,*.egg-info}
    bash bin/build-cuda-ext.sh build/patched-src/flash_attn-2.8.3 FLASH_ATTENTION_FORCE_BUILD
    cp build/patched-src/flash_attn-2.8.3/dist/flash_attn-*.whl build/wheels/
    # Trigger devenv's reinstall task to pick up the new wheels without
    # requiring a full ``devenv up`` cycle.
    wheels=(build/wheels/*.whl)
    uv pip install --reinstall --no-deps "$${wheels[@]}"
    echo "=== verifying imports ==="
    uv run --no-sync python -c "import causal_conv1d; import mamba_ssm; import flash_attn; \
        print('causal_conv1d', causal_conv1d.__version__); \
        print('mamba_ssm', mamba_ssm.__version__); \
        print('flash_attn', flash_attn.__version__)"

# Aggressive flash-attn build tuned for the 64-core / 125 GB / 6×4090
# box. See bin/build-flash-attn-aggressive.sh for full rationale.
#
# Highlights:
#   MAX_JOBS=16, NVCC_THREADS=1  → ~40 GB cicc RAM peak, ~25 min wall
#                                  (vs 2-3 h at MAX_JOBS=4)
#   Temporary 16 GB swapfile     → safety net for memory spikes; an
#                                  overshoot spills to disk instead of
#                                  triggering a kernel deadlock
#   Devenv stack stopped         → frees postgres/qdrant/gateway RAM
#                                  during the build, restarted on exit
#   Stray-cicc pre-check         → refuses to start if an earlier build
#                                  left D-state children behind
#   Stall monitor                → aborts after 15 min of no progress
#   Cleanup trap                 → always reaps children + drops swap,
#                                  even on Ctrl-C / failure
#
# After a successful build, the wheel lands in ``build/wheels/`` and is
# auto-reinstalled by devenv's ``aegir:cuda-ext-reinstall`` task on
# every ``devenv up`` / ``devenv shell`` — no need to run a separate
# install command.
#
# Override: ``MAX_JOBS=8 just build-flash-attn`` etc.
build-flash-attn:
    bash bin/build-flash-attn-aggressive.sh
