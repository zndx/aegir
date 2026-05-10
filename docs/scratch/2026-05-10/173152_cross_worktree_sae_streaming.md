# Cross-worktree SAE-stream development

Pattern for splitting development across two git worktrees of the
aegir repo:

- **systems worktree** (this one): runs P5 GRPO/RLVR training. Writes
  SAE feature records to `/raid/checkpoints/p5/`.
- **UI worktree**: runs the FastAPI gateway + Vite dev server.
  Subscribes to the gateway's SSE stream of those SAE records, renders
  morphism-style visualizations in the React UI.

## Shared state

Both worktrees share the underlying filesystem, devenv venv, postgres,
qdrant, and the `/raid/checkpoints/p5/` directory. Coordination
needed:

| Resource | Constraint | Default |
|---|---|---|
| Postgres | Single instance per host | `:5555` (devenv `services.postgres`) |
| Qdrant | Single instance per host | `:6355/:6356` |
| Gateway port | Pick one worktree to run it | `:8091` |
| Vite dev port | Pick one worktree to run it | `:5173` |
| `/raid/checkpoints/p5/` | Single writer (training side) | `aegir.config.P5Cfg.output_dir` |

## Worktree role detection

`bin/detect-worktree-role.sh` prints `primary` or `secondary` based on
whether `.git` is a directory (primary) or a file pointing at the
shared `.git/worktrees/<name>/` (secondary, created by
`git worktree add`). Both `just` and `devenv` consume this:

- **`just whoami`** prints the current role + key shared-state
  defaults. First thing to run when picking up a worktree session.
- **`just p5-train`** refuses in secondary worktrees (writes to
  `/raid/checkpoints/p5/`, shared state). `--dry-run` is exempt;
  `ALLOW_SECONDARY=1` overrides if you've set a distinct
  `AEGIR_P5_OUTPUT_DIR`.
- **`just gateway`** refuses in secondary worktrees by default
  (port collision). `ALLOW_SECONDARY=1` + a distinct
  `AEGIR_GATEWAY_PORT` override.
- **`just ui-dev`** has no role guard — secondary is its natural home;
  Vite picks an alternate port automatically if `:5173` is taken.
- **`devenv up`** in a secondary worktree skips `services.postgres`,
  the `processes` block (qdrant + gateway + vite-dev), and prints a
  hint at shell entry. The secondary worktree connects to the
  primary's services via `localhost:<port>`.

The env var `AEGIR_WORKTREE_ROLE` overrides the script's output, so
exotic layouts (a third worktree, a CI harness mocking secondary)
can opt in/out explicitly.

Convention: the **primary** checkout owns shared state (training,
postgres, qdrant, gateway). The **secondary** checkout is a satellite
(UI dev, ad-hoc scripts) that connects to the primary's services.

## Per-worktree role mapping

| Worktree | Role | Runs |
|---|---|---|
| `aegir` (this) | primary | `devenv up` (postgres + qdrant + gateway), `just p5-train` |
| `ae-ui-dev` | secondary | `just ui-dev` (Vite dev), connects to primary's `:8091` |

Run the gateway + Vite dev in **either** worktree if you really want
(set `ALLOW_SECONDARY=1`); the convention above is the default to
avoid accidental dual-launch.

## SAE-stream pipeline

```
   p5_train.py (systems worktree)
        │
        │  every step (cadence: sae_live_spill_every_n_steps)
        ▼
   /raid/checkpoints/p5/sae_features.live.jsonl   (run-root tail)
        │
        │  every 50 steps on save
        ▼
   /raid/checkpoints/p5/checkpoint-N/sae_features.jsonl   (snapshots)
        │
        │  filesystem read
        ▼
   gateway/app.py: GET /api/p5/sae/stream  (SSE, prefers live tail)
        │
        │  EventSource subscription
        ▼
   UI worktree: ui/src/...     (Bokeh / D3 / etc.)
```

### SSE event types

`GET /api/p5/sae/stream` emits four event types:

- **default `data:` events** — one per JSON record. Body is the
  `SAELogRecord` dataclass dict: `{step, token_index, layer_index,
  top_feature_indices, top_feature_activations, reconstruction_loss}`.
- **`event: source`** — emitted whenever the stream switches between
  `live` (preferred) and `snapshot:checkpoint-N` (fallback). Body:
  `{source: "live" | "snapshot:...", path: "..."}`. The UI should
  reset its tail-position state on this event.
- **`event: checkpoint`** — emitted when the snapshot fallback rolls
  to a new `checkpoint-N` directory. Body: `{step, checkpoint}`.
- **`event: idle`** — emitted while neither a live log nor any
  checkpoint exists yet. Body: `{reason: "..."}`.
- **`event: heartbeat`** — every ~10 s to keep proxies awake. Body:
  `{}`.

### Quick sanity check from the UI worktree

```bash
# (1) Train side — start a 9B-local run in this systems worktree.
just p5-train

# (2) UI side — in the other worktree:
just gateway        # serves on :8091
curl -N http://localhost:8091/api/p5/runs
curl -N http://localhost:8091/api/p5/sae/stream | head -20
```

## What the UI gets to visualize

Per record:

- **`top_feature_indices`** + **`top_feature_activations`** — which
  K SAE features fired hardest at this layer for this token (default
  K=16). The morphism story: input bytes → SAE features → ontology
  term selection → output bytes.
- **`reconstruction_loss`** — `||x - SAE(x)||²` on the residual
  stream. A spike here flags an under-budgeted moment (the L0=50
  sparsity dropped a concept that mattered for this composition).
- **`layer_index`** — 0 / 8 / 16 / 24 / 32 / 39 (Qwen3.5-9B has 40
  layers), so the UI can split early/middle/late activity.

The brief's morphism reading: input ontology corpus → SAE features
common to gate-passing compositions are interpretable as the policy's
"vocabulary" for ontology-term selection. The UI surfaces this in
near-real-time so a user can correlate a bad reward with the features
that fired during that generation.
