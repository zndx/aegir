{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
    just
    awscli2
    cmake
    conftest
    d2
    dbmate
    git
    gh
    graphviz
    grpcurl
    imagemagick
    jq
    mdbook
    mdbook-d2
    mdbook-katex
    mdbook-mermaid
    ninja
    nodejs_22      # PGlite + Vite
    pnpm           # UI package manager
    protobuf
    presenterm
    flatbuffers
    sops           # air-gap secret decryption (bin/bootstrap-secrets.sh)
    # zarf         # uncomment when pkgs.zarf is upstreamed; for now
    #              # follow cybersec's convention: fetch via curl in CI or vendor
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv.enable = true;
    uv.sync.enable = true;
    venv.enable = true;
  };

  # ── Disk routing for model weights + caches ──────────────────
  #
  # The system drive (/) is space-constrained (~15 GB free at
  # last audit). All large artifacts MUST land on /raid:
  #
  #   - HuggingFace model + dataset cache → /raid/cache/huggingface
  #     (also reached via the ~/.cache/huggingface symlink for
  #     tools that don't honor HF_HOME)
  #   - P5 checkpoints + LoRA adapters + GRPO logs →
  #     /raid/checkpoints/p5/  (CheckpointConfig default in
  #     src/aegir/rl/checkpointing.py)
  #   - Benchmark datasets (SOTAB, GitTables) →
  #     /raid/datasets/ (existing convention, see Justfile)
  env = {
    HF_HOME = "/raid/cache/huggingface";
    HF_HUB_CACHE = "/raid/cache/huggingface/hub";
    HUGGINGFACE_HUB_CACHE = "/raid/cache/huggingface/hub";
    TRANSFORMERS_CACHE = "/raid/cache/huggingface/hub";
  };

  # ── PostgreSQL 16 with pgvector ─────────────────────────────
  #
  # Port 5555 chosen to avoid conflicts with sibling projects on the same
  # dev box:
  #   atelier 5533 / 5440-CAI, gaius 5444, cybersec 5438, signals 5455
  # pgvector is load-bearing for M2+ (column-embedding similarity search);
  # provisioned here so migrations can `CREATE EXTENSION vector` unconditionally.
  services.postgres = {
    enable = true;
    package = pkgs.postgresql_16;
    port = 5555;
    listen_addresses = "127.0.0.1";
    extensions = extensions: [ extensions.pgvector ];
    initialDatabases = [{ name = "aegir"; }];
  };

  # ── Process management ──────────────────────────────────────
  #
  # ``devenv up`` starts every process below. Python services call
  # ``aegir.config.load_config()`` which reads HOCON with live env
  # substitution — dotenv.enable injects env vars at shell entry, so no
  # materialized config file is strictly required in the devenv loop.
  # ``just resolve-config`` materializes build/config/aegir.{env,json}
  # for conftest / BDD / CI.
  processes = {
    # Qdrant vector store — HTTP 6355, gRPC 6356.
    # Storage under DEVENV_STATE survives devenv restarts but is excluded
    # from git.  Provisioned empty in M1; populated in M2.
    qdrant = {
      exec = ''
        mkdir -p $DEVENV_STATE/qdrant
        QDRANT__STORAGE__STORAGE_PATH=$DEVENV_STATE/qdrant/storage \
        QDRANT__SERVICE__HTTP_PORT=6355 \
        QDRANT__SERVICE__GRPC_PORT=6356 \
        ${pkgs.qdrant}/bin/qdrant
      '';
      process-compose.readiness_probe = {
        http_get = {
          host = "localhost";
          port = 6355;
          path = "/healthz";
        };
        initial_delay_seconds = 2;
        period_seconds = 2;
        failure_threshold = 15;
      };
    };

    # Gateway (FastAPI). Applies migrations inline before launching
    # uvicorn so the gateway never sees an un-bootstrapped schema.
    # Folding bootstrap into the startup command (instead of a
    # one-shot dependency) dodges process-compose sequencing races.
    gateway = {
      exec = ''
        uv run --no-sync python -m aegir.db.bootstrap && \
        exec uv run --no-sync python -m aegir.gateway
      '';
      process-compose = {
        depends_on.postgres.condition = "process_healthy";
        readiness_probe = {
          http_get = {
            host = "localhost";
            port = 8091;
            path = "/api/health";
          };
          initial_delay_seconds = 3;
          period_seconds = 2;
          failure_threshold = 30;
        };
      };
    };

    # Vite dev server for the React UI.  Starts after gateway so the
    # /api proxy (vite.config.ts) has something to talk to.
    vite-dev = {
      exec = "cd ui && pnpm install --silent --prefer-offline && pnpm dev";
      process-compose.depends_on.gateway.condition = "process_healthy";
    };
  };

  # ── Patched CUDA-extension reinstall ─────────────────────────
  #
  # ``uv.sync.enable = true`` runs ``uv sync`` on every ``devenv up`` /
  # ``devenv shell``, which replaces our ABI-patched flash-attn +
  # mamba-ssm wheels with the PyPI versions (which are built against
  # cxx11abi=TRUE while torch cu124 is cxx11abi=FALSE — see
  # bin/build-flash-attn-aggressive.sh). This task runs *after* the
  # sync and reinstalls the patched wheels from build/wheels/ when
  # they exist. No-op on a fresh clone (the user runs ``just
  # build-flash-attn`` once to populate build/wheels/).
  tasks."aegir:cuda-ext-reinstall" = {
    description = "Reinstall ABI-patched CUDA extensions from build/wheels/";
    after = [ "devenv:python:uv" ];
    before = [ "devenv:enterShell" ];
    # Delegate to the canonical Justfile recipe so the patched-wheel
    # restore step has one implementation. ``just sync`` calls the
    # same recipe directly when the user runs uv sync outside of
    # devenv up / devenv shell.
    exec = ''
      cd "$DEVENV_ROOT"
      just _restore-patched-wheels
    '';
  };

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;
}
