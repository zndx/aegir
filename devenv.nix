{ pkgs, lib, config, inputs, ... }:

let
  # Worktree role detection. ``git worktree add <path>`` creates a
  # *secondary* checkout whose ``.git`` is a file pointing back at
  # the primary's ``.git/worktrees/<name>/`` rather than a real
  # directory. The convention this file enforces:
  #
  #   primary    → owns shared state; runs postgres + qdrant +
  #                gateway + vite-dev under ``devenv up``.
  #   secondary  → satellite; skips those services entirely so two
  #                concurrent ``devenv up`` invocations don't bind
  #                colliding ports / corrupt the same DB cluster.
  #
  # ``bin/detect-worktree-role.sh`` (committed) prints the role at
  # the shell. devenv reads ``AEGIR_WORKTREE_ROLE`` from the
  # environment so the user (or ``.envrc``) can set it once at
  # entry. Defaults to "primary" when unset, preserving existing
  # single-worktree behavior.
  worktreeRole = lib.maybeEnv "AEGIR_WORKTREE_ROLE" "primary";
  isPrimary = worktreeRole == "primary";
in {
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

  dotenv.enable = true;

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv.enable = true;
    uv.sync.enable = true;
    venv.enable = true;
  };

  languages.rust = {
    enable = true;
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
    # Re-export the resolved role into the shell so ``just``
    # recipes and downstream scripts read the same value devenv
    # used to gate services.
    AEGIR_WORKTREE_ROLE = worktreeRole;
  };

  enterShell = ''
    if [ "${worktreeRole}" != "primary" ]; then
      echo "[aegir] devenv: worktree role = ${worktreeRole}; skipping postgres/qdrant/gateway/vite-dev."
      echo "        These services run only in the primary checkout to avoid port collisions."
      echo "        Connect to the primary's services from here, or set AEGIR_WORKTREE_ROLE=primary"
      echo "        to override (and accept the responsibility for collision-free port choice)."
    fi
  '';

  # ── PostgreSQL 16 with pgvector ─────────────────────────────
  #
  # Port 5555 chosen to avoid conflicts with sibling projects on the same
  # dev box:
  #   atelier 5533 / 5440-CAI, gaius 5444, cybersec 5438, signals 5455
  # pgvector is load-bearing for M2+ (column-embedding similarity search);
  # provisioned here so migrations can `CREATE EXTENSION vector` unconditionally.
  services.postgres = {
    enable = isPrimary;
    package = pkgs.postgresql_16;
    port = 5555;
    listen_addresses = "127.0.0.1";
    extensions = extensions: [
      extensions.age
      extensions.pg_cron
      extensions.pgvector
    ];
    # age + pg_cron must be preloaded at server start (pg_cron requires it; age
    # loads its shared lib so cypher() resolves). Matches the working sibling
    # configs (signals "age,pg_cron", gaius "pg_cron,age"). pgvector needs no preload.
    settings.shared_preload_libraries = "age,pg_cron";
    initialDatabases = [{ name = "aegir"; }];
  };

  services.caddy = {
    enable = true;
    # One origin for the control-plane UI, fronting both governance surfaces:
    #   /api/atlas/* -> the real Apache Atlas v2 service (forked, AGE backend) on :21000
    #   /api/*       -> the aegir gateway (control API + extended OpenLineage variant)
    # handle blocks are first-match, so the more specific /api/atlas/* precedes /api/*.
    virtualHosts."http://localhost:8080".extraConfig = ''
      handle /api/atlas/* {
        reverse_proxy 127.0.0.1:21000
      }
      handle /api/* {
        reverse_proxy 127.0.0.1:8091
      }
    '';
  };

  # JDK 21 + Maven to build/run the forked Apache Atlas (AGE backend on aegir_hx).
  languages.java = {
    enable = true;
    jdk.package = pkgs.jdk21;
    maven.enable = true;
  };

  # ── Process management ──────────────────────────────────────
  #
  # ``devenv up`` starts every process below. Python services call
  # ``aegir.config.load_config()`` which reads HOCON with live env
  # substitution — dotenv.enable injects env vars at shell entry, so no
  # materialized config file is strictly required in the devenv loop.
  # ``just resolve-config`` materializes build/config/aegir.{env,json}
  # for conftest / BDD / CI.
  # Service processes are skipped entirely in secondary worktrees.
  # Each worktree gets its own ``processes`` set; here we either
  # populate it (primary) or leave it empty (secondary).
  processes = lib.mkIf isPrimary {
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

    # Graph-substrate bootstrap (one-shot). After Postgres is healthy, applies
    # migrations/ — CREATE EXTENSION age + create_graph(aegir_hx) + labels — so the
    # provenance graph is ensured on every ``devenv up``, implicitly, for everyone.
    # NOTHING depends on it (no process-compose sequencing race); it fails loud if
    # AGE is absent from the build (no fallback). Idempotent via schema_migrations.
    db-bootstrap = {
      exec = "uv run --no-sync python -m aegir.db.bootstrap";
      process-compose = {
        depends_on.postgres.condition = "process_healthy";
        availability.restart = "no";
      };
    };

    # Apache Atlas (forked, AGE backend) on the SHARED aegir_hx graph — :21000.
    # Atlas v2 entities live in aegir_hx alongside our provenance (one graph, no
    # duplication). Gated on db-bootstrap so aegir_hx + its labels exist first.
    # Requires a one-time build: `devenv tasks run atlas:build`.
    atlas = {
      exec = ''
        set -e
        ATLAS_DIR="$PWD/components/atlas"
        ATLAS_WEBAPP="$ATLAS_DIR/webapp/target/atlas-webapp-3.0.0-SNAPSHOT"
        ATLAS_CONF="$PWD/config/atlas"
        ATLAS_HOME="$PWD/.devenv/atlas"
        mkdir -p "$ATLAS_HOME/data" "$ATLAS_HOME/logs" "$ATLAS_HOME/conf"
        ln -sfn "$ATLAS_DIR/addons/models" "$ATLAS_HOME/models"   # type-system bootstrap source
        cp -n "$ATLAS_CONF/users-credentials.properties" "$ATLAS_HOME/conf/" 2>/dev/null || true
        cp -n "$ATLAS_CONF/atlas-simple-authz-policy.json" "$ATLAS_HOME/conf/" 2>/dev/null || true
        if [ ! -d "$ATLAS_WEBAPP/WEB-INF" ]; then
          echo "Atlas webapp not built. Run: devenv tasks run atlas:build"; exit 1
        fi
        echo "Starting Atlas on http://localhost:21000 (AGE backend -> aegir_hx)..."
        exec java \
          -Datlas.home="$ATLAS_HOME" -Datlas.conf="$ATLAS_CONF" \
          -Datlas.log.dir="$ATLAS_HOME/logs" -Datlas.log.file=application \
          -Datlas.data="$ATLAS_HOME/data" \
          -Dlogback.configurationFile="$ATLAS_DIR/distro/src/conf/atlas-logback.xml" \
          -Datlas.graphdb.backend=org.apache.atlas.repository.graphdb.age.AtlasAgeGraphDatabase \
          -Djava.net.preferIPv4Stack=true \
          --add-opens java.base/java.lang=ALL-UNNAMED \
          --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
          --add-opens java.base/java.io=ALL-UNNAMED \
          --add-opens java.base/java.net=ALL-UNNAMED \
          --add-opens java.base/java.util=ALL-UNNAMED \
          --add-opens java.base/java.util.concurrent=ALL-UNNAMED \
          --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
          --add-opens java.base/sun.security.action=ALL-UNNAMED \
          --add-opens java.security.jgss/sun.security.krb5=ALL-UNNAMED \
          -server -Xmx1024m \
          -cp "$ATLAS_CONF:$ATLAS_WEBAPP/WEB-INF/classes:$ATLAS_WEBAPP/WEB-INF/lib/*" \
          org.apache.atlas.Atlas -app "$ATLAS_WEBAPP" -port 21000
      '';
      process-compose = {
        depends_on = {
          postgres.condition = "process_healthy";
          db-bootstrap.condition = "process_completed_successfully";
        };
        readiness_probe = {
          exec.command = "curl -sf http://127.0.0.1:21000/api/atlas/admin/status";
          initial_delay_seconds = 15;
          period_seconds = 10;
          timeout_seconds = 5;
          failure_threshold = 30;
        };
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
  # Build the forked Apache Atlas webapp with the AGE graph provider (one-time, ~minutes).
  tasks."atlas:build" = {
    description = "Build the forked Apache Atlas webapp (AGE backend)";
    exec = ''
      cd "$DEVENV_ROOT/components/atlas"
      mkdir -p webapp/target/api/v2/apidocs/ui
      mvn package -pl webapp -am -Dmaven.test.skip=true -DskipUTs=true \
        -DGRAPH-PROVIDER=age -Dcheckstyle.skip=true -DskipEnunciate=true \
        --no-transfer-progress
    '';
  };

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
