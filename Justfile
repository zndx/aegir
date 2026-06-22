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

# ── Worktree role (primary | secondary) ───────────────────────
#
# git worktree-add creates linked checkouts whose ``.git`` is a
# *file* pointing back to the original repo's ``.git/worktrees/<name>/``
# rather than a real directory. ``bin/detect-worktree-role.sh``
# reads that distinction and prints ``primary`` (original) or
# ``secondary`` (linked).
#
# Convention: the **primary** checkout owns shared state (devenv
# services, /raid checkpoint writes). Any **secondary** checkout
# is a satellite — read-mostly UI / scripting work that connects
# to the primary's running services. Recipes that bind ports or
# touch shared state branch on this value:
#
#   * services that conflict on port (gateway, vite-dev) refuse
#     to start in a secondary worktree unless ``ALLOW_SECONDARY=1``
#     is set, with a hint pointing at the primary;
#   * services that write shared state (p5-train) refuse to run
#     in a secondary worktree unconditionally.
#
# devenv.nix reads the same value via ``AEGIR_WORKTREE_ROLE``
# and gates ``services.postgres`` / ``services.qdrant`` on it
# so ``devenv up`` in a secondary checkout doesn't try to bind
# the primary's ports.
worktree_role := `if [ -n "${AEGIR_WORKTREE_ROLE:-}" ]; then echo "$AEGIR_WORKTREE_ROLE"; else bin/detect-worktree-role.sh 2>/dev/null || echo primary; fi`

default:
    @just --list

# Print this checkout's worktree role + the relevant shared-state
# defaults. Use this for a quick sanity-check before launching
# services; if it says ``secondary`` and you didn't expect that,
# investigate before running ``just gateway`` etc.
whoami:
    @echo "worktree_role     = {{worktree_role}}"
    @echo "git toplevel      = $(git rev-parse --show-toplevel)"
    @echo "common git dir    = $(git rev-parse --git-common-dir)"
    @echo "branch            = $(git rev-parse --abbrev-ref HEAD)"
    @echo "p5 output_dir     = ${AEGIR_P5_OUTPUT_DIR:-/raid/checkpoints/p5}"
    @echo "gateway port      = ${AEGIR_GATEWAY_PORT:-8091}"
    @echo "all worktrees:"
    @git worktree list | sed 's/^/  /'

# ── Dependency sync ───────────────────────────────────────────
#
# ``just sync`` is the single entry point that reliably brings the
# venv to a consistent state. It runs ``uv sync`` and then re-applies
# the patched flash-attn / mamba-ssm / causal-conv1d wheels under
# ``build/wheels/`` — those wheels are ABI-patched per CLAUDE.md and
# are clobbered by every plain ``uv sync``. Idempotent; safe to run
# any number of times.
#
# Use ``just sync`` after pulling new commits, after editing
# pyproject.toml, or any time the venv feels stale. The devenv
# post-uv-sync hook (in devenv.nix) calls the same restore logic so
# ``devenv up`` / ``devenv shell`` users do not need to remember.
#
# If ``build/wheels/`` is empty, ``just sync`` skips the restore step
# and prints a hint. To populate it, run ``just cuda-deps`` (or
# ``just build-flash-attn`` for the aggressive 25-min path).
sync:
    uv sync
    @just _restore-patched-wheels

# Hidden recipe — used by ``just sync`` and by devenv's
# aegir:cuda-ext-reinstall task. Single canonical implementation.
# Works whether invoked from inside ``devenv shell`` (VIRTUAL_ENV is
# already set) or directly (we fall back to the devenv-managed venv
# under .devenv/state/venv).
_restore-patched-wheels:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "$(git rev-parse --show-toplevel)"
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        if [ -d ".devenv/state/venv" ]; then
            export VIRTUAL_ENV="$PWD/.devenv/state/venv"
        else
            echo "[aegir] no VIRTUAL_ENV set and no .devenv/state/venv found — run 'devenv up' first or activate the venv"
            exit 1
        fi
    fi
    shopt -s nullglob
    wheels=(build/wheels/*.whl)
    if [ "${#wheels[@]}" -eq 0 ]; then
        echo "[aegir] no patched wheels under build/wheels/ — run 'just cuda-deps' to populate"
        exit 0
    fi
    echo "[aegir] restoring patched CUDA extensions: ${wheels[*]##*/}"
    uv pip install --reinstall --no-deps "${wheels[@]}" >/dev/null

check-ontology-schema:
    uv run --no-sync python scripts/check_ontology_schema.py

# Meta-harness inc-2b: run the H₀ single-file harness over gap topics (frozen Grok
# + ContractGate/HermiT) via the candidate filesystem. Needs the JVM libs
# (DeepOnto/HermiT) + cuda driver libs. coverage-run defaults to the canonical v1 ground.
mediate-h0 topics="124" coverage_run="/raid/checkpoints/aegir-artifacts/coverage_v1/043d7dcc185245c8" max_iters="6":
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs:$(pwd)/build/cuda-driver-libs \
        uv run --no-sync python scripts/mediate_h0.py \
        --coverage-run {{coverage_run}} --topics {{topics}} --max-iters {{max_iters}}

# Build the lineup-navigable KB projection at build/dev/{current,scratch,archive} from the
# three Data Products (ontology, relational, content). Regenerable + gitignored (current/ is
# what we KNOW; the corpora submodule is what we SHARE). content/topics project from on-disk
# corpus/coverage runs when present (set AEGIR_CORPUS_RUN / AEGIR_COVERAGE_RUN to point them).
kb-build:
    uv run --no-sync python -m aegir.lineup build

# Run the full content-first pipeline end-to-end as one Metaflow flow (harvest → derive → membrane/
# promote → realized DDL → content-first chapters → verify → lineup → Atlas). The Metaflow service
# plane (service+UI+MinIO on RKE2) comes up via `devenv up`; this just runs the flow, traced to the
# OTel collector (→ NiFi). One command. e.g. `just metaflow --n-docs 8`.
metaflow *ARGS:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4327 \
    uv run --no-sync python -m aegir.flows.semantic_corpus_flow run {{ARGS}}

# Path A: continue-pretrain RWKV-7 ± augmentation across α-arms, eval general+relational, write the ledger
# (the data-value-isolation experiment as one CI-gated command). Arms train one-per-GPU; pass `--max-workers
# N` to parallelize. e.g. `AEGIR_METAFLOW_MODE=local just train-path-a --alphas 0,0.02 --max-workers 2`.
train-path-a *ARGS:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4327 \
    uv run --no-sync python -m aegir.flows.path_a_training_flow run {{ARGS}}

# Manually (re)deploy the Metaflow service plane to RKE2 (idempotent; normally devenv does this on `up`).
metaflow-up:
    bash scripts/metaflow/db-setup.sh
    bash scripts/metaflow/bootstrap.sh

# Run a lineup KB upkeep op by hand (normally driven by pg_cron → scheduled_tasks →
# the in-gateway processor). op = upkeep (age past-quarter scratch → archive) | snapshot
# (current/ → archive snapshot) | reproject. `install-cron` registers the pg_cron jobs.
kb-maintain op="upkeep":
    uv run --no-sync python -m aegir.lineup maintain {{op}}

# Freeze current/ into a namespaced, self-contained archive snapshot before a regen, so the
# present lineup isn't lost. `just kb-snapshot --key 2026Q3` → build/dev/archive/2026Q3/ (ids +
# wikilinks prefixed, coexists with a regen-latest current, survives kb-build). Default key =
# calendar quarter. Provenance in _manifest.json makes it reproducible.
kb-snapshot *args:
    uv run --no-sync python -m aegir.lineup maintain snapshot {{args}}

# SHARE: re-publish the ontology Data Product (catalog incl. broader hierarchy + SKOS + DDL)
# from the live SoT → the corpora (sdg-corpora) submodule. Schema-CI gated. No args = dry
# materialize + diff (review first); `just kb-sync --commit` commits corpora locally;
# `just kb-sync --push` publishes to zndx/sdg-corpora (the external share).
kb-sync *args:
    uv run --no-sync python -m aegir.lineup sync {{args}}

# SHARE-Docs Phase A: render the lineup KB projection → a browsable mdbook (collections × lens
# pivot + the cross-linked ontology/relational/content graph, wikilinks lowered to page links).
# `just kb-render` renders + builds → build/dev/lineup-book/book/index.html. Run `just kb-build` first.
kb-render *args:
    uv run --no-sync python scripts/render_lineup_mdbook.py --build {{args}}

# ── capability/gRPC engine ────────────────────────────────────
#
# The Aegir capability engine (minimal mirror of the Gaius pattern): workloads request an
# inference CAPABILITY (e.g. "instruct") over gRPC; the engine alone owns capability→model
# selection and is the ONLY thing that talks to vLLM (strict layering — no vLLM URL ever
# leaves the engine). vLLM runs in a dedicated cu129 venv (see engine/config.py VLLM_PYTHON);
# the cuda-driver-libs prefix unmasks libcuda (Nix). Override the model via AEGIR_INSTRUCT_MODEL.
engine-serve:
    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs uv run --no-sync python -m aegir.engine.server

# Smoke-test the engine: a capability request returns text via gRPC (engine→vLLM internally).
engine-ping prompt="In one sentence, what is a foreign key?":
    uv run --no-sync python -c "from aegir.engine.client import complete; print(complete('''{{prompt}}''', capability='instruct', max_tokens=64, temperature=0.3))"

# ── mdbook documentation ──────────────────────────────────────
#
# The book lives at ``docs/current/`` (standard ``mdbook init``
# layout): config at ``docs/current/book.toml``, sources under
# ``docs/current/src/``, theme at ``docs/current/theme/``, generated
# HTML at ``docs/current/book/``. Per-session scratch + archived
# chapters live outside this book root and are excluded from the
# build.
#
# These recipes work identically from any worktree because they
# pin the absolute book path rather than relying on a relative
# cwd.
docs-build:
    mdbook build docs/current

docs-serve:
    mdbook serve docs/current --open

docs-clean:
    rm -rf docs/current/book docs/current/src/d2

# Run the SDG runtime verifier on a composition. P1b-α scope:
# R_A + R_B + R_C; R_D stubbed at 0.0 until P1b-β.
#   just aegir-verify path/to/composition.json
#   echo '{"templates":[...]}' | just aegir-verify -
aegir-verify composition='-':
    uv run --no-sync python scripts/aegir-verify.py --composition {{composition}}

# Bootstrap the surgical LD_LIBRARY_PATH symlinks under build/jvm-libs.
# Re-run after a reboot if /tmp was your old jvm-libs location, or any
# time the system libs at /usr/lib/x86_64-linux-gnu move. Idempotent.
setup-libs:
    bash scripts/setup_jvm_env.sh && ls build/jvm-libs/

# Run the offline DeepOnto pass over a candidate catalog JSON,
# populating is_complex/verbal_template/mean_verbal_length per
# template. Bootstraps the surgical LD_LIBRARY_PATH the JVM
# needs; see scripts/setup_jvm_env.sh.
build-catalog input output:
    bash -c 'source scripts/setup_jvm_env.sh && uv run --no-sync python scripts/build_catalog.py {{input}} {{output}}'

# Fit T_I on the pinned input corpus (held-out SchemaPile +
# FinePDFs-lab) and compute the null distribution over the
# combined catalog. Caches T_I to src/aegir/ontology/catalog/T_I.pkl
# and writes null_stats.json there.
build-topic-model *args:
    bash -c 'source scripts/setup_jvm_env.sh && uv run --no-sync python scripts/build_topic_model.py {{args}}'

# C1 verifier validation: build labeled test set, score each
# ontology end-to-end, sweep aggregation weights {a, b, c} for
# AUC, report results.
c1-validate:
    uv run --no-sync python scripts/build_test_set.py
    bash -c 'source scripts/setup_jvm_env.sh && uv run --no-sync python scripts/tune_verifier_weights.py'

# P4 smoke test: single GRPO group iteration over the locked
# verifier — validates reward-signal propagation, determinism,
# and quality discrimination without committing to actual model
# training. See `scripts/p4_smoke_test.py` for the design rationale.
p4-smoke:
    bash -c 'source scripts/setup_jvm_env.sh && uv run --no-sync python scripts/p4_smoke_test.py'

# P5 launcher: GRPO/RLVR training of SAE-Res-Qwen3.5 against the
# locked SDG verifier.
#
# ``just p5-train`` does the right thing on the Tinybox out of the
# box: Qwen3.5-9B-Base FSDP-sharded across 2 of the 6 GPUs, 2 SAE
# observation layers on rank 0, the C1-locked verifier in the GRPO
# reward, and the per-step live SAE JSONL flush that the gateway's
# SSE stream subscribes to. ``--dry-run`` short-circuits before any
# GPU work; ``--resume`` picks up the latest checkpoint with the
# strict drift check; see ``scripts/p5_train.py --help`` for the
# full flag set and ``--policy-preset`` for scale/sparsity overrides
# (advanced — only when you need the 27B-FSDP LambdaLabs target or a
# different L0 / single-GPU regime).
#
# Routing under the hood:
#   ``--dry-run``       → plain ``uv run python`` (no GPU work)
#   ``*-local-*`` preset → plain ``uv run python`` (single GPU)
#   ``*-fsdp-*``  preset → ``accelerate launch --use_fsdp`` with
#                          ``num_processes`` from the preset
#                          (2 for 9b-fsdp, 6 for 27b-fsdp).
p5-train *args:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/setup_jvm_env.sh
    # Refuse in secondary worktrees: p5-train writes to a shared
    # /raid/checkpoints/p5/ tree (per-checkpoint dirs + the live SAE
    # JSONL the gateway streams). Two concurrent runs would clobber
    # each other's checkpoints and corrupt the live tail. ALLOW_SECONDARY=1
    # overrides for the rare deliberate dual-train scenario.
    if [ "{{worktree_role}}" = "secondary" ] && [ "${ALLOW_SECONDARY:-0}" != "1" ] \
            && [[ " {{args}} " != *" --dry-run "* ]]; then
        cat <<EOF >&2
    [aegir] worktree role = secondary; refusing p5-train.
    p5-train writes to ${AEGIR_P5_OUTPUT_DIR:-/raid/checkpoints/p5}/, shared
    state across worktrees. Run in the primary worktree, or set
    ALLOW_SECONDARY=1 with a distinct AEGIR_P5_OUTPUT_DIR if you really
    need a parallel run.
    EOF
        exit 2
    fi
    # Reduce fragmentation: the 24 GB envelope on a single 4090 is
    # tight at 9B + LoRA + 2 SAE layers + 8-way GRPO generation
    # activations + KV cache. ``expandable_segments:True`` lets
    # PyTorch's caching allocator grow segments rather than fragment
    # into many fixed-size blocks; observed OOMs at "21.9 GB allocated,
    # 564 MiB reserved-but-unallocated" recover with this flag.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    # Detect parallelism + num_processes from the policy preset.
    # ``*-fsdp-*`` presets route through accelerate launch with the
    # preset's matching num_processes; everything else (default
    # 9b-local-l0-50, ``--dry-run`` anywhere) runs single-process.
    is_single_gpu=true
    num_processes=1
    if [[ " {{args}} " == *" --policy-preset 9b-fsdp-"* ]]; then
        is_single_gpu=false
        num_processes=2
    elif [[ " {{args}} " == *" --policy-preset 27b-fsdp-"* ]]; then
        is_single_gpu=false
        num_processes=6
    fi
    if [[ " {{args}} " == *" --dry-run "* ]]; then
        uv run --no-sync python scripts/p5_train.py {{args}}
    elif $is_single_gpu; then
        uv run --no-sync python scripts/p5_train.py {{args}}
    else
        # Per-rank error files: torch-elastic captures each rank's
        # exception as JSON so multi-process failures surface
        # properly (rather than the generic <NO_OTHER_FAILURES>).
        # Logs land under /raid/checkpoints/p5/launch-logs/.
        log_dir="/raid/checkpoints/p5/launch-logs"
        mkdir -p "$log_dir"
        export TORCHELASTIC_ERROR_FILE="$log_dir/rank_\${RANK}_error.json"
        # FSDP knobs:
        #  * sharding_strategy FULL_SHARD     — params, grads, optimizer all sharded
        #  * cpu_ram_efficient_loading true   — only rank 0 reads weights from disk;
        #                                       other ranks init on meta and sync.
        #                                       Without this, every rank tries to
        #                                       materialize the full 54 GB model on
        #                                       its single 24 GB GPU before sharding,
        #                                       OOMing at FSDP _move_module_to_device.
        #  * sync_module_states true          — broadcast init from rank 0
        #  * transformer_layer_cls_to_wrap
        #      Qwen3_5DecoderLayer            — the layer FSDP wraps as a unit.
        #                                       TRANSFORMER_BASED_WRAP needs this
        #                                       hint to find the right granularity.
        #  * use_orig_params true             — needed for LoRA-on-FSDP compatibility
        #                                       (peft attaches new params after FSDP
        #                                       wrap; orig_params=True keeps them
        #                                       trainable).
        # No --mixed_precision flag: the model is already loaded
        # as bf16 by ``load_policy`` (torch_dtype="bfloat16") and
        # all LoRA params are cast to bf16 in ``policy.py``. With
        # accelerate's MixedPrecisionPolicy, FSDP would try to
        # unshard params at a "mp param dtype" that diverges from
        # the original ``lora_A.weight.dtype`` peft reads at the
        # call site, producing the ``mat1/mat2 dtype mismatch``
        # observed in the GRPO generation forward pass. Letting the
        # native bf16 dtype propagate end-to-end keeps both sides
        # of every matmul aligned.
        uv run --no-sync accelerate launch \
            --num_processes "$num_processes" \
            --use_fsdp \
            --fsdp_sharding_strategy FULL_SHARD \
            --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            --fsdp_cpu_ram_efficient_loading true \
            --fsdp_sync_module_states true \
            --fsdp_use_orig_params true \
            --log_dir "$log_dir" \
            --tee 3 \
            scripts/p5_train.py {{args}}
    fi

# P5 rejection-sampling SFT data generator. Reads the C1-locked verifier
# + the constrained-decode schema, samples K compositions per prompt
# variation, scores each with verify(), and writes high-reward samples
# to /raid/checkpoints/p5/sft-corpus/rejection_samples.jsonl.
#
# Single-GPU inference (9B-bf16 fits in 24 GB with 384-token KV cache
# and the lmfe prefix_allowed_tokens_fn state). The other 5 GPUs stay
# free for parallel work (UI session, secondary worktree experiments).
p5-rejection-sample *args:
    bash -c 'source scripts/setup_jvm_env.sh && uv run --no-sync python scripts/p5_rejection_sample.py {{args}}'

# P5 SFT trainer. Consumes the rejection-sampled corpus and produces a
# LoRA adapter that warm-starts GRPO. The output directory feeds
# ``just p5-train --init-checkpoint <dir>``. FSDP across 2 GPUs by
# default; matches the 9b-fsdp-l0-50 envelope of p5-train.
p5-sft *args:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/setup_jvm_env.sh
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    uv run --no-sync accelerate launch \
        --num_processes 2 \
        --use_fsdp \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
        --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
        --fsdp_cpu_ram_efficient_loading true \
        --fsdp_sync_module_states true \
        --fsdp_use_orig_params true \
        scripts/p5_sft.py {{args}}

bdd-0: check-ontology-schema
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

get-schemapile:
    uv run --no-sync python scripts/download_schemapile.py

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
#                                                  docs/current/src/pretraining.md)

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
# ontology-grounded pretraining (docs/current/src/pretraining.md).
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
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{worktree_role}}" = "secondary" ] && [ "${ALLOW_SECONDARY:-0}" != "1" ]; then
        cat <<EOF >&2
    [aegir] worktree role = secondary; refusing to bind gateway port.
    The primary worktree owns the gateway. Either:
      - run "just gateway" in the primary worktree, or
      - export ALLOW_SECONDARY=1 to run a satellite gateway here
        (set AEGIR_GATEWAY_PORT to avoid the primary's :8091).
    EOF
        exit 2
    fi
    uv run --no-sync python -m aegir.gateway

ui-dev:
    cd ui && pnpm install --silent --prefer-offline && pnpm dev

ui-build:
    cd ui && pnpm install --silent --prefer-offline && pnpm build

# Live HoloViews viz server (topology B): a dedicated bokeh server behind the gateway's reverse proxy
# at /viz/*, embedded into React via <PanelView> (server_document). BOKEH_RESOURCES=server keeps every
# resource same-origin → air-gapped. The gateway proxies /viz (prod: nginx; dev: vite proxy on :5173).
viz-serve:
    BOKEH_RESOURCES=server LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs \
      uv run --no-sync bokeh serve src/aegir/viz/lineup_app.py src/aegir/viz/runs_app.py \
      src/aegir/viz/sweeps_app.py src/aegir/viz/reward_app.py src/aegir/viz/provenance_app.py \
      --prefix /viz --port 5006 --allow-websocket-origin='*'

# Lightweight, repeatable exercise+verify of the leaderboard viz: a smoke training run produces a real
# run, then scripts/verify_viz.py builds+renders every plot through runs.py's HoloViews builders (the
# runs_app path). Browser gate (air-gap, no JS errors): `just viz-serve` + open /leaderboards.
verify-viz:
    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs uv run --no-sync python train.py --smoke-test --model-size tiny --epochs 3
    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs uv run --no-sync python scripts/verify_viz.py

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
