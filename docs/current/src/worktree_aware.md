# Worktree Aware Development

Ægir's dev tooling is **worktree aware**: `just` recipes and `devenv up`
detect whether the current checkout is the *primary* one or a *secondary*
linked worktree (created by `git worktree add`), and gate shared-state
services accordingly. This lets you split a working session across two
checkouts — e.g. systems work in one, UI iteration in another — without
port collisions, double-launched databases, or two copies of a training
run fighting over the same checkpoint directory.

## Why bother

Plain `git clone` gives you one working directory. When you need to look
at two branches simultaneously — say, low-level model work on `trunk` and
UI iteration on `feat/leaderboard-redesign` — the usual options are all
expensive:

- `git stash` + `git checkout` repeatedly: loses focus, scrambles your
  editor state, gates the two contexts behind serial discipline.
- A second `git clone`: doubles the disk usage, doubles the dependency
  install, doesn't share `.git/objects` so fetches are duplicated, two
  separate venvs to keep in sync.
- A pair of containers: heavyweight, hostile to GPU passthrough, and
  the in-host tooling has to be reproduced inside each.

`git worktree add <path> <branch>` is the under-used answer:

```bash
# In the primary checkout
git worktree add ../ae-ui-dev rch/ui-dev
```

This creates a second working directory at `../ae-ui-dev` checked out to
`rch/ui-dev`, sharing `.git/objects` with the primary. Two editor windows,
two shells, one repo. The catch: both checkouts share the host. Run
`devenv up` in both and they fight for `:5555` (Postgres), `:6355`
(Qdrant), `:8091` (gateway), and the on-disk Postgres data dir.

Ægir's worktree-aware tooling resolves that without the user having to
remember which checkout is "the one with the database in it".

## Detection

`bin/detect-worktree-role.sh` reads two filesystem signals:

```d2
direction: right

primary: {
  shape: rectangle
  label: "primary worktree\n.git/ is a directory"
  style: { fill: "#dff2dd"; stroke-width: 2 }
}

secondary: {
  shape: rectangle
  label: "secondary worktree\n.git is a file pointing at\n<primary>/.git/worktrees/<name>/"
  style: { fill: "#fdf2dd"; stroke-width: 2 }
}

shared: {
  shape: cylinder
  label: ".git/objects (shared)\n.git/refs (shared)"
  style: { fill: "#eee" }
}

primary -> shared
secondary -> shared
```

The script's logic is two `test` statements:

```bash
if [ -d "$top/.git" ]; then
    echo primary
elif [ -f "$top/.git" ]; then
    echo secondary
fi
```

That's it. No state, no daemons, no config files. The check runs in <5 ms
and is called from both the `Justfile` (parse-time) and `devenv.nix`
(via the `AEGIR_WORKTREE_ROLE` env var that the primary's `enterShell`
sets for downstream tools).

## What's gated

| Service / recipe | Primary | Secondary | Override |
|---|---|---|---|
| `services.postgres` (devenv) | enabled | **disabled** | n/a |
| `processes.qdrant` (devenv) | enabled | **disabled** | n/a |
| `processes.gateway` (devenv) | enabled | **disabled** | n/a |
| `processes.vite-dev` (devenv) | enabled | **disabled** | n/a |
| `just gateway` | runs | **refuses** | `ALLOW_SECONDARY=1` + `AEGIR_GATEWAY_PORT=…` |
| `just p5-train` | runs | **refuses** | `ALLOW_SECONDARY=1` + `AEGIR_P5_OUTPUT_DIR=…` |
| `just ui-dev` | runs | runs | (no guard — Vite picks an alt port) |
| `just whoami` | runs | runs | (diagnostic only) |
| `just sync` / `bdd-*` / etc. | runs | runs | (no shared-state writes) |

The table shows the headline four; in fact the **entire primary
`processes` set** — including the Metaflow service plane (RKE2
bootstrap, port-forwards, UI), Atlas, MinIO, and the OTel collector —
is gated behind `lib.mkIf isPrimary` in `devenv.nix`, so a secondary
`devenv up` starts no shared-state services at all.

Secondary worktrees connect to the primary's services via `localhost:<port>`.
The filesystem path `/raid/checkpoints/p5/` is shared — the primary writes,
the secondary's gateway (if running, via `ALLOW_SECONDARY`) reads, the
secondary's UI subscribes to the SSE stream regardless of which gateway it
talks to.

## Workflow

```bash
# ── In the primary checkout (e.g. systems work) ───────────────
just whoami                 # → primary
devenv up                   # postgres, qdrant, gateway, vite-dev (and the
                            # rest of the primary service plane) start
just p5-train               # 9B-local GRPO/RLVR training
                            # writes /raid/checkpoints/p5/sae_features.live.jsonl
```

```bash
# ── In the secondary checkout (e.g. UI iteration) ─────────────
git worktree add ../ae-ui-dev rch/ui-dev
cd ../ae-ui-dev
just whoami                 # → secondary
devenv up                   # skips services with a hint; safe to call
just ui-dev                 # Vite dev server (auto-picks free port)
                            # subscribes to primary's :8091 gateway
                            # SSE: GET /api/p5/sae/stream
```

The UI sees feature activations within seconds of each GRPO step — see
[Cross-worktree SAE streaming](./worktree_aware/sae_streaming.md) for
the data path.

## Overriding the role

Set `AEGIR_WORKTREE_ROLE` to override the detection script. Useful when:

- A CI harness wants to mock secondary behavior in the primary checkout.
- You've added a third worktree and want it to behave as primary
  (and accepted responsibility for picking a non-colliding port set).
- The secondary checkout *is* the systems checkout for a session
  (you flipped roles deliberately).

```bash
AEGIR_WORKTREE_ROLE=secondary just whoami     # force-secondary
AEGIR_WORKTREE_ROLE=primary devenv up         # force-primary
```

`devenv` reads the env var via `lib.maybeEnv "AEGIR_WORKTREE_ROLE" "primary"`,
so the primary default is preserved when the var is unset.

## When to bypass the guards

Each guard has an explicit override path so you're never blocked, just
forced to be deliberate:

- `ALLOW_SECONDARY=1 AEGIR_P5_OUTPUT_DIR=/raid/checkpoints/p5-experiment-2 \
   just p5-train --policy-preset 9b-local-l0-100` — runs a parallel
  training run from the secondary checkout against a distinct output
  directory.
- `ALLOW_SECONDARY=1 AEGIR_GATEWAY_PORT=8092 just gateway` — runs a
  satellite gateway from the secondary checkout on a distinct port (e.g.
  for testing a UI build against an isolated backend).

The pattern is: the recipe refuses by default to surface the conflict,
and the env vars give you the explicit knobs needed to make the override
collision-free.

## Adding new worktree-aware recipes

When adding a recipe that binds a port or writes shared state, branch on
`{{worktree_role}}` early:

```just
my-new-recipe:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{worktree_role}}" = "secondary" ] && [ "${ALLOW_SECONDARY:-0}" != "1" ]; then
        echo "[aegir] worktree role = secondary; refusing my-new-recipe." >&2
        echo "        Run in primary, or ALLOW_SECONDARY=1 with a distinct OUTPUT_DIR." >&2
        exit 2
    fi
    # ... actual recipe body ...
```

The diagnostic message should name (a) the conflicting resource and
(b) the env var(s) that disambiguate an override.

For pure-compute recipes (no port, no shared write) — e.g. the BDD
suite, smoke tests, schema validators — leave them unguarded. They're
safe to run in any worktree concurrently.

## See also

- [Cross-worktree SAE streaming](./worktree_aware/sae_streaming.md) — the
  filesystem-and-SSE pipeline that lets a UI worktree observe a training
  run in the primary worktree in near-real-time.
- `bin/detect-worktree-role.sh` — the detection script.
- `Justfile` — the `worktree_role` parse-time variable and per-recipe
  guards.
- `devenv.nix` — the `worktreeRole` let-binding and `lib.mkIf isPrimary`
  service gates.
