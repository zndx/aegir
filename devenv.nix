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
  # CUDA toolchain for compiling the fused RWKV-7 training kernel (rwkv7_clampw) via torch
  # cpp_extension.load. nix-native (nvcc 12.8 + gcc-14.3 + libstdc++ all on nix glibc 2.42) so it is
  # CONSISTENT with the nix python — the system CUDA-12.4 nvcc/cicc clashes (older system glibc) and
  # cannot be used in-process. Trainer sets CUDA_HOME/CC/CXX/CUDAHOSTCXX from AEGIR_CUDA_* (below) for its
  # own compile; we deliberately do NOT set CC/CXX globally (that would break rust/maturin builds).
  # Re-import the pinned nixpkgs with cudaForwardCompat OFF. devenv's nixpkgs defaults it ON, which pulls
  # `cuda_compat` (a forward-compat shim with no redistributable source here) → build failure. The system
  # driver (build/cuda-driver-libs) already supports CUDA 12.8, so the compat shim is unneeded. (Verified:
  # the merged toolchain builds + the fused kernel compiles+runs under it.)
  cudaPkgs = import pkgs.path { inherit (pkgs) system; config = { allowUnfree = true; cudaForwardCompat = false; }; };
  cuda = cudaPkgs.cudaPackages_12_8;
  cudaMerged = cudaPkgs.symlinkJoin {
    name = "aegir-cuda-12.8";
    paths = [ cuda.cuda_nvcc cuda.cuda_cudart cuda.cuda_cccl ];
    # torch cpp_extension probes $CUDA_HOME/lib64; nix uses lib → alias it.
    postBuild = ''
      if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then ln -s lib "$out/lib64"; fi
    '';
  };
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
    maturin        # build polyglot-sql (Rust/PyO3) Python extension
    sops           # air-gap secret decryption (bin/bootstrap-secrets.sh)
    minio-client   # `mc` — S3 client for bucket management on Signals' RustFS (:9010)
    tilt           # K8s deploy of the Metaflow service plane (`tilt ci`, mirrors gaius)
    kubectl        # RKE2 control for the Metaflow service plane
    kubernetes-helm # metaflow-tools chart (via tilt helm_remote)
    gettext        # `envsubst` — render Endpoints/values with the runtime pg port
    cudaMerged     # nix-native CUDA 12.8 (nvcc+cudart+cccl) for the fused RWKV-7 kernel compile
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
    # LOAD-BEARING: without these extras uv sync PRUNES the URL-pinned flash-attn /
    # mamba-ssm / causal-conv1d wheels (pyproject [tool.uv.sources]; patched-wheel
    # era closed 2026-07-19 — see CLAUDE.md "CUDA Extension Notes").
    uv.sync.extras = [ "flash" "mamba" ];
    # --inexact: the shell-entry auto-sync must be ADDITIVE — exact mode would prune
    # the task-built polyglot-sql editable install (deliberately not a uv source) on
    # every pyproject change. The deliberate full-consistency path stays `just sync`
    # (exact). First two flags = devenv's defaults, repeated because setting the
    # option replaces them.
    uv.sync.arguments = [ "--frozen" "--no-install-workspace" "--inexact" ];
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
    # Metaflow (the orchestration substrate; see config/metaflow/, infra/tilt/). Its S3
    # datastore is Signals' RustFS on 127.0.0.1:9010 — aegir runs NO object store of its
    # own (the devenv MinIO, marked insecure upstream, was retired 2026-09-05). Never
    # export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY project-wide: the ambient default
    # credential chain leaks into every S3 client; consumers read AEGIR_RUSTFS_* explicitly
    # (config/metaflow/*.json carries the Metaflow copy).
    METAFLOW_HOME = "${config.devenv.root}/.metaflow";
    AEGIR_RUSTFS_ENDPOINT = "127.0.0.1:9010";
    AEGIR_RUSTFS_ACCESS_KEY = "rustfsadmin";
    AEGIR_RUSTFS_SECRET_KEY = "rustfsadmin";
    # nix-native CUDA toolchain paths for the fused RWKV-7 kernel compile (the trainer reads these and sets
    # CUDA_HOME/CC/CXX/CUDAHOSTCXX for its own cpp_extension.load — see scripts/continue_pretrain_rwkv7.py).
    AEGIR_CUDA_HOME = "${cudaMerged}";
    AEGIR_CUDA_CCBIN = "${cuda.backendStdenv.cc}/bin";
  };

  enterShell = ''
    export KUBECONFIG="$HOME/.config/kube/rke2.yaml"   # RKE2 for the Metaflow service plane
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
    # "*" so in-cluster Metaflow pods can reach the host postgres via the aegir-postgres
    # Endpoints (mirrors gaius). pg_hba is opened to the cluster CIDR below.
    listen_addresses = "*";
    extensions = extensions: [
      extensions.age
      extensions.pg_cron
      extensions.pgvector
    ];
    # age + pg_cron must be preloaded at server start (pg_cron requires it; age
    # loads its shared lib so cypher() resolves). Matches the working sibling
    # configs (signals "age,pg_cron", gaius "pg_cron,age"). pgvector needs no preload.
    settings.shared_preload_libraries = "age,pg_cron";
    initialDatabases = [{ name = "aegir"; } { name = "metaflow"; }];
    # NOTE: in-cluster pod access also needs pg_hba to allow the RKE2 pod CIDR (calico 10.42.0.0/16,
    # md5). devenv's default hba + listen_addresses="*" matches the working gaius setup; if the
    # metaflow-service pod can't reach postgres at deploy, append a `host all all 10.42.0.0/16 md5` rule.
  };

  services.varnish = {
    enable = true;
    # 608x is Ranger territory (6080 HTTP, 6085 Tomcat shutdown socket) —
    # the varnish lattice lives in the 609x decade: gaius 6091, signals 6092,
    # aegir 6093, atelier 6094.
    listen = "127.0.0.1:6093";
    # Federated menu pattern (gaius precedent): only the *_origin route is
    # cached — ttl+grace serves the waffle instantly while a background
    # fetch refreshes. Backend is the gateway (:8091), which serves the
    # raw roster; everything else passes through untouched.
    vcl = ''
      vcl 4.1;

      backend aegir_gateway {
        .host = "127.0.0.1";
        .port = "8091";
        .connect_timeout = 2s;
        .first_byte_timeout = 120s;
      }

      sub vcl_recv {
        if (req.url ~ "^/api/aegir/v1/federation/surfaces_origin") {
          return (hash);
        }
        return (pass);
      }

      sub vcl_backend_response {
        if (bereq.url ~ "^/api/aegir/v1/federation/surfaces_origin") {
          if (beresp.status >= 400) {
            # A background refresh that fails must not displace the good
            # stale object; a foreground error must not stick in cache.
            if (bereq.is_bgfetch) {
              return (abandon);
            }
            set beresp.ttl = 1s;
            set beresp.grace = 0s;
            set beresp.uncacheable = true;
          } else {
            set beresp.ttl = 60s;
            set beresp.grace = 6h;
          }
        }
      }
    '';
  };

  # (services.minio retired 2026-09-05: the Metaflow S3 datastore is Signals' RustFS,
  #  127.0.0.1:9010, bucket aegir-metaflow — see the env block above.)

  # ── OpenTelemetry collector — the Step→OTel→NiFi spine (mirrors gaius) ────────
  # Flow @traced_step spans → OTLP :4327/:4328 → forwarded to NiFi ListenOTLP :4329 (flow-viz) + debug.
  # Aegir ports (4327/4328/4329, prom 8890) disambiguate from gaius (4317/4318/4319, 8889).
  services.opentelemetry-collector = lib.mkIf isPrimary {
    enable = true;
    package = pkgs.opentelemetry-collector-contrib;
    settings = {
      receivers.otlp.protocols = {
        grpc.endpoint = "0.0.0.0:4327";
        http.endpoint = "0.0.0.0:4328";
      };
      processors.batch = { timeout = "5s"; send_batch_size = 1000; };
      exporters = {
        debug.verbosity = "basic";
        prometheus = { endpoint = "0.0.0.0:8890"; namespace = "aegir"; };
        otlphttp = { endpoint = "http://localhost:4329"; tls.insecure = true; };  # → NiFi ListenOTLP
      };
      service.pipelines = {
        traces = { receivers = ["otlp"]; processors = ["batch"]; exporters = ["debug" "otlphttp"]; };
        metrics = { receivers = ["otlp"]; processors = ["batch"]; exporters = ["prometheus"]; };
      };
    };
  };

  services.caddy = {
    enable = true;
    # One origin for the control-plane UI, fronting both governance surfaces:
    #   /api/atlas/* -> the real Apache Atlas v2 service (forked, AGE backend) on :21000
    #   /api/*       -> the aegir gateway (control API + extended OpenLineage variant)
    # handle blocks are first-match, so the more specific /api/atlas/* precedes /api/*.
    virtualHosts.":8080".extraConfig = ''
      handle /api/atlas/* {
        reverse_proxy 127.0.0.1:21000
      }
      handle /viz/* {
        reverse_proxy 127.0.0.1:5006
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
        python3 bin/reclaim-port.py qdrant 6355 6356
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

    # ── Metaflow service plane on RKE2 (mirrors gaius; ns aegir-metaflow) ──────
    # db-setup (one-shot) → bootstrap (tilt ci, one-shot) → port-forwards (NodePort 30181 + UI).
    # ui is opt-in (`devenv processes up metaflow-ui`). Disable all via DISABLE_METAFLOW=true.
    metaflow-db-setup = {
      exec = "exec ${config.devenv.root}/scripts/metaflow/db-setup.sh";
      process-compose = {
        depends_on.postgres.condition = "process_healthy";
        availability.restart = "no";
      };
    };
    metaflow-bootstrap = {
      exec = "exec ${config.devenv.root}/scripts/metaflow/bootstrap.sh";
      process-compose = {
        depends_on = {
          postgres.condition = "process_healthy";
          metaflow-db-setup.condition = "process_completed_successfully";
        };
        availability.restart = "no";
      };
    };
    metaflow-port-forwards = {
      exec = "exec ${config.devenv.root}/scripts/metaflow/port-forwards.sh";
      process-compose = {
        depends_on.metaflow-bootstrap.condition = "process_completed_successfully";
        availability.restart = "always";
      };
    };
    metaflow-ui = {
      exec = "exec ${config.devenv.root}/scripts/metaflow/ui.sh";
      process-compose = {
        depends_on.metaflow-db-setup.condition = "process_completed_successfully";
        disabled = true;
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
        python3 bin/reclaim-port.py atlas 21000
        ATLAS_DIR="$PWD/components/atlas"
        ATLAS_WEBAPP="$ATLAS_DIR/webapp/target/atlas-webapp-3.0.0-SNAPSHOT"
        ATLAS_CONF="$PWD/config/atlas"
        ATLAS_HOME="$PWD/.devenv/atlas"
        mkdir -p "$ATLAS_HOME/data" "$ATLAS_HOME/logs" "$ATLAS_HOME/conf"
        ln -sfn "$ATLAS_DIR/addons/models" "$ATLAS_HOME/models"   # type-system bootstrap source
        # Materialize Atlas config with the LIVE PGPORT — Atlas reads the file and does
        # NOT reliably interpolate ${env:...}, so substitute the real port at start.
        sed "s|localhost:[0-9]*/aegir|localhost:$PGPORT/aegir|" "$ATLAS_CONF/atlas-application.properties" > "$ATLAS_HOME/conf/atlas-application.properties"
        cp -f "$ATLAS_CONF/users-credentials.properties" "$ATLAS_HOME/conf/" 2>/dev/null || true
        cp -f "$ATLAS_CONF/atlas-simple-authz-policy.json" "$ATLAS_HOME/conf/" 2>/dev/null || true
        if [ ! -d "$ATLAS_WEBAPP/WEB-INF" ]; then
          echo "Atlas webapp not built. Run: devenv tasks run atlas:build"; exit 1
        fi
        # Atlas's HikariPool fail-fasts (no retry) if Postgres isn't accepting yet —
        # wait for the live port so a startup race can't wedge Atlas in a 503 loop.
        for _i in $(seq 1 90); do pg_isready -h localhost -p "$PGPORT" -q && break; sleep 1; done
        echo "Starting Atlas on http://localhost:21000 (AGE backend -> aegir_hx, pg :$PGPORT)..."
        exec java \
          -Datlas.home="$ATLAS_HOME" -Datlas.conf="$ATLAS_HOME/conf" \
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
          -cp "$ATLAS_HOME/conf:$ATLAS_WEBAPP/WEB-INF/classes:$ATLAS_WEBAPP/WEB-INF/lib/*" \
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

    # Gateway (FastAPI) — the one application's backend (serves /api/* + the static
    # React bundle; vite-dev proxies to it). Applies migrations inline before launching
    # uvicorn so it never sees an un-bootstrapped schema, and projects the lineup KB
    # (build/dev/) so /api/kb serves a populated lineup — both folded into startup
    # (NOT separate processes) per the established pattern. The lineup projection is
    # non-blocking: if it fails the gateway still serves (the rest of the app), and
    # /api/kb 404s until `just kb-build` succeeds.
    # PORT RECLAIM FIRST, build LAST-before-exec: the reclaim step evicts orphans/zombies
    # holding :8091 (a manually-launched gateway that outlives its session), and its
    # position AHEAD of the expensive KB build means a bind conflict can never again
    # crash-loop the wipe/rebuild under a stale server (2026-08-07: 1,130 restarts,
    # lineup "non-existent" — see docs/scratch/2026-08-07/181500_*).
    gateway = {
      exec = ''
        python3 bin/reclaim-port.py gateway 8091
        # Canonical federation identity — the gateway builds the waffle
        # roster in-process (self row + peers), so it needs the same
        # advertise host as the engine or FQDN detection leaks the WAN
        # reverse-DNS name.
        export AEGIR_ADVERTISE_HOST="''${AEGIR_ADVERTISE_HOST:-''${SIGNALS_ADVERTISE_HOST:-tinybox.dev.vista.zndx.org}}"
        uv run --no-sync python -m aegir.db.bootstrap && \
        { uv run --no-sync python -m aegir.lineup build || echo "[gateway] lineup projection failed — /api/kb will 404 until 'just kb-build' succeeds"; } && \
        AEGIR_LINEUP_UPKEEP=1 AEGIR_GATEWAY_RELOAD=1 exec uv run --no-sync python -m aegir.gateway
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

    # Live HoloViews / Bokeh server (topology B) — lineup chords, runs, sweeps.
    # Vite and the gateway reverse-proxy /viz → :5006. Without this process the
    # lineup Lexicon chord 404s `/viz/static/js/bokeh-gl.min.js`.
    viz = {
      exec = ''
        python3 bin/reclaim-port.py viz 5006
        export BOKEH_RESOURCES=server
        export LD_LIBRARY_PATH="${config.devenv.root}/build/cuda-driver-libs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        exec uv run --no-sync bokeh serve \
          src/aegir/viz/lineup_app.py src/aegir/viz/runs_app.py \
          src/aegir/viz/sweeps_app.py src/aegir/viz/reward_app.py \
          src/aegir/viz/provenance_app.py \
          --prefix /viz --port 5006 --allow-websocket-origin='*'
      '';
      process-compose = {
        readiness_probe = {
          http_get = {
            host = "localhost";
            port = 5006;
            path = "/viz/static/js/bokeh.min.js";
          };
          initial_delay_seconds = 3;
          period_seconds = 3;
          failure_threshold = 40;
        };
      };
    };

    # Vite dev server for the React UI.  Starts after gateway so the
    # /api proxy (vite.config.ts) has something to talk to.
    vite-dev = {
      exec = ''
        python3 bin/reclaim-port.py vite-dev 5173
        cd ui && pnpm install --silent --prefer-offline && pnpm dev
      '';
      process-compose.depends_on.gateway.condition = "process_healthy";
    };

    # Capability / lattice engine (:50151) — multi-service gRPC for signals.target.
    # Full-stack doctrine: devenv up owns product UI *and* the federation engine.
    # Does not wait for vLLM SERVING; readiness is gRPC listen + Status.
    capability-engine = {
      exec = ''
        python3 bin/reclaim-port.py capability-engine 50151 || true
        export LD_LIBRARY_PATH="${config.devenv.root}/build/cuda-driver-libs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export AEGIR_ENGINE_PORT="''${AEGIR_ENGINE_PORT:-50151}"
        # Canonical federation identity: without this advertise_host() falls
        # back to FQDN detection, which picks up the WAN reverse-DNS name
        # (customer.*.isp.starlink.com) — unresolvable over WARP/off-LAN.
        export AEGIR_ADVERTISE_HOST="''${AEGIR_ADVERTISE_HOST:-''${SIGNALS_ADVERTISE_HOST:-tinybox.dev.vista.zndx.org}}"
        exec uv run --no-sync python -m aegir.engine.server
      '';
      process-compose = {
        readiness_probe = {
          exec.command = "bash -c '</dev/tcp/127.0.0.1/50151'";
          initial_delay_seconds = 3;
          period_seconds = 2;
          failure_threshold = 60;
        };
      };
    };
  };

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

  # Build the polyglot-sql Python extension (Rust/PyO3 via maturin) into the venv.
  # Like the patched CUDA wheels, this is task-built (NOT a uv source) so ``uv sync``
  # never rebuilds the Rust crate. One-time / on submodule bump (~minutes).
  tasks."polyglot:build" = {
    description = "Build polyglot-sql (maturin develop --release) into the devenv venv";
    exec = ''
      cd "$DEVENV_ROOT/components/polyglot/crates/polyglot-sql-python"
      maturin develop --release
    '';
  };

  # (The aegir:cuda-ext-reinstall task that restored ABI-0 patched wheels from
  # build/wheels/ after every sync was REMOVED 2026-08-07 — the patched-wheel era is
  # closed; uv.sync.extras above keeps the good upstream wheels installed instead.)

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;
}
