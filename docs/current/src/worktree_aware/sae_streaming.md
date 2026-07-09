# Cross-worktree SAE streaming

The motivating use case for [Worktree Aware Development](../worktree_aware.md):
a P5 GRPO/RLVR training run (the long-horizon Signals-M4 apparatus — see
[RLVR for ontology generation](../ontology/rlvr.md)) executes in the
primary worktree while a React UI runs in a secondary worktree, and the
UI observes SAE feature activations from the running training process in
near-real-time.

## Pipeline

```d2
direction: right

train: {
  shape: rectangle
  label: "p5_train.py\n(primary worktree)"
  style: { fill: "#dff2dd" }
}

live_jsonl: {
  shape: page
  label: "sae_features.live.jsonl\n(run root, appended every step)"
  style: { fill: "#fff8dd" }
}

snap_jsonl: {
  shape: page
  label: "checkpoint-N/sae_features.jsonl\n(snapshot, on save)"
  style: { fill: "#fff8dd" }
}

gateway: {
  shape: rectangle
  label: "FastAPI gateway\nGET /api/p5/sae/stream\n(SSE, prefers live tail)"
  style: { fill: "#ddeeff" }
}

ui: {
  shape: rectangle
  label: "React UI\n(secondary worktree)\nEventSource subscriber"
  style: { fill: "#ddeeff" }
}

train -> live_jsonl: "SidecarCallback.on_step_end\n(every N steps)"
train -> snap_jsonl: "SidecarCallback.on_save\n(every 50 steps)"
live_jsonl -> gateway: "preferred"
snap_jsonl -> gateway: "fallback"
gateway -> ui: "text/event-stream"
```

The two JSONL files participate at different cadences:

- **`sae_features.live.jsonl`** at the run root (default
  `/raid/checkpoints/p5/sae_features.live.jsonl`) — appended every
  `sae_live_spill_every_n_steps` GRPO steps (default: 1) by
  `SidecarCallback.on_step_end`. Records produced between spills are
  cleared from the in-memory buffer on each spill, so the live tail and
  the per-checkpoint snapshots are mutually exclusive (no
  double-counting).
- **`checkpoint-N/sae_features.jsonl`** under each checkpoint dir —
  point-in-time snapshot written on every `Trainer` save event (every
  50 steps by default). Primary source for post-hoc analysis; the
  fallback for the SSE stream when no live tail exists yet (i.e.
  resume-from-checkpoint where the trainer has saved but hasn't yet
  reached the first live-spill cadence tick).

## Endpoints

```text
GET /api/p5/runs
```

Walks `cfg.p5.output_dir` for `checkpoint-N/` subdirs. Returns one row
per checkpoint with the `aegir_metadata.json` sidecar payload plus a
count of SAE feature records spilled to that checkpoint:

```json
{
  "rows": [
    {
      "step": 50,
      "checkpoint": "checkpoint-50",
      "metadata": { "run_id": "…", "catalog_version": "0.7.0-combined", … },
      "n_sae_records": 320,
      "checkpoint_dir": "/raid/checkpoints/p5/checkpoint-50"
    }
  ],
  "count": 1,
  "p5_dir": "/raid/checkpoints/p5",
  "p5_dir_exists": true
}
```

```text
GET /api/p5/sae/stream
```

Server-sent events stream of SAE feature records. Polls `cfg.p5.output_dir`
every 0.5 s. Prefers the live JSONL when present; falls back to the
latest checkpoint's snapshot otherwise. Five event kinds:

| Event | Body shape | When |
|---|---|---|
| (default) | `SAELogRecord` JSON | one per new line in the active source |
| `event: source` | `{source: "live" \| "snapshot:…", path}` | active source switched |
| `event: checkpoint` | `{step, checkpoint}` | snapshot fallback rolled to a new checkpoint dir |
| `event: idle` | `{reason}` | no live log and no checkpoints exist yet |
| `event: heartbeat` | `{}` | every ~10 s, keeps proxies awake |

### Record fields

Each default `data:` event carries one `SAELogRecord` dict:

```json
{
  "step": 42,
  "token_index": 8,
  "layer_index": 16,
  "top_feature_indices":     [3201, 14872, 5891, …],
  "top_feature_activations": [4.21, 3.87,  3.04, …],
  "reconstruction_loss":     0.0142
}
```

- `top_feature_indices` + `top_feature_activations` — the K SAE features
  that fired hardest at this layer for this token (default K=16).
- `reconstruction_loss` — `||x − SAE(x)||²` on the residual stream. A
  spike here flags an under-budgeted moment (the L0 sparsity dropped a
  concept that mattered for this composition).
- `layer_index` — which transformer layer's residual produced this
  record. The hooked layers are chosen by `default_layers_to_hook(num_layers,
  n)` (`aegir.rl.policy`), `n` evenly-spaced indices via `step = num_layers // n`.
  `n` is the `--sae-num-layers` flag, default **2** — so Qwen3.5-9B-Base (40
  layers) hooks `[0, 20]` and the 27B base (64 layers) hooks `[0, 32]`. Raising
  `--sae-num-layers 4` spreads them wider (`[0, 10, 20, 30]` on the 9B, `[0, 16,
  32, 48]` on the 27B).

## Smoke test

From the secondary worktree, while the primary runs `just p5-train`:

```bash
# In the primary checkout
just p5-train --policy-preset 9b-local-l0-50

# In the secondary checkout (a different shell)
curl http://localhost:8091/api/p5/runs | jq '.count, .rows[0]'
curl -N http://localhost:8091/api/p5/sae/stream | head -20
```

The `-N` flag on `curl` disables output buffering; without it, SSE
events queue in 4 KB chunks and you won't see them until enough land.

## What the UI gets to visualize

The [concept brief](../ontology/concept_brief.md)'s morphism reading:
**input bytes → SAE features → ontology term selection → output
bytes**. The UI surfaces this in near-real-time
so a user can correlate a bad reward with the features that fired
during the failing generation:

- A heatmap of `top_feature_activations` over `(layer, time)` shows
  which layers' SAE dictionaries dominated each generation.
- A scatter of `reconstruction_loss` vs. final `R` (verifier reward,
  recorded separately in `grpo_metrics.jsonl`) tests whether the L0
  sparsity budget is dropping concepts that matter — a negative
  correlation motivates a post-hoc L0=50 ablation.
- Sustained activation of a small feature subset across gate-passing
  generations is the post-hoc interpretability claim: those features
  *are* the policy's "vocabulary" for ontology-term selection in the
  morphism.

## Configuration

`aegir.config.P5Cfg` (mirroring `aegir.rl.checkpointing.CheckpointConfig`):

| Field | Default | Override |
|---|---|---|
| `output_dir` | `/raid/checkpoints/p5` | `AEGIR_P5_OUTPUT_DIR` |
| `sae_log_filename` | `sae_features.jsonl` | (HOCON `aegir.p5.sae_log_filename`) |
| `sae_live_log_filename` | `sae_features.live.jsonl` | (HOCON `aegir.p5.sae_live_log_filename`) |
| `metadata_filename` | `aegir_metadata.json` | (HOCON `aegir.p5.metadata_filename`) |

`aegir.rl.checkpointing.CheckpointConfig`:

| Field | Default | Notes |
|---|---|---|
| `sae_live_spill_every_n_steps` | `1` | Cadence of `SidecarCallback.on_step_end` flush |

The primary and secondary worktrees see the same `output_dir` because
both inherit `AEGIR_P5_OUTPUT_DIR` (or the default) — the `/raid` path
is host-shared. No additional IPC, no additional ports, no additional
state.
