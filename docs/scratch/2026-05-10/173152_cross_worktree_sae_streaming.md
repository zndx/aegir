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

Run the gateway + Vite dev in **either** worktree (it's the same
recipe via `just gateway` and `just ui-dev`). The training process is
always the systems worktree.

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
