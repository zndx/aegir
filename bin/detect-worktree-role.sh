#!/usr/bin/env bash
# Print this checkout's role in a multi-worktree layout:
#
#   primary    — the original working directory of the repo. ``.git``
#                is a regular directory.
#   secondary  — a linked worktree created by ``git worktree add``.
#                ``.git`` is a file whose contents start with
#                ``gitdir: ../<common>/.git/worktrees/<name>``.
#
# Convention: the *primary* checkout owns shared state (devenv
# postgres + qdrant data, p5 training that writes to /raid). Any
# *secondary* checkout is a read-mostly satellite (UI dev, gateway
# subscriptions, ad-hoc scripts) that connects to the primary's
# services rather than spinning up its own.
#
# Usage:
#   role=$(bin/detect-worktree-role.sh)
#   case $role in primary) ... ;; secondary) ... ;; esac
#
# devenv.nix reads ``AEGIR_WORKTREE_ROLE`` (the env-var form of this
# script's output) via ``lib.maybeEnv`` and gates ``services.*`` on
# the role so ``devenv up`` in a secondary checkout doesn't try to
# bind colliding postgres / qdrant ports.

set -euo pipefail

# Resolve the toplevel; bail out gracefully if not inside a git repo.
top=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$top" ]; then
    echo primary  # not a git repo — single-checkout fallback
    exit 0
fi

if [ -d "$top/.git" ]; then
    echo primary
elif [ -f "$top/.git" ]; then
    echo secondary
else
    # Submodule / bare-clone / unusual layout — treat as primary so
    # we don't accidentally skip services the user does want.
    echo primary
fi
