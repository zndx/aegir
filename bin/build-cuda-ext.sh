#!/bin/bash
# Build one patched CUDA extension (causal-conv1d / mamba-ssm / flash-attn)
# into a wheel under build/wheels/, using ``env -i`` to strip the nix
# GCC-15 toolchain in favor of system GCC-11, matching torch cu124's
# CXX11 ABI (``_GLIBCXX_USE_CXX11_ABI=0``).
#
# Usage:
#   bin/build-cuda-ext.sh <source-dir> <force-build-env-var>
#
# Example:
#   bin/build-cuda-ext.sh /raid/repos/aegir-patched/causal_conv1d-1.6.1 CAUSAL_CONV1D_FORCE_BUILD
#
# Output: wheel(s) under ``{source-dir}/dist/`` — caller is responsible
# for copying them to ``build/wheels/`` and ``uv pip install --no-deps``'ing them.
#
# Background on the recipe: CLAUDE.md + docs/scratch/2026-03-28/010808_deps_smoke_train.md

set -euo pipefail

SRC_DIR="${1:?source dir required}"
FORCE_VAR="${2:?FORCE_BUILD env var name required, e.g. FLASH_ATTENTION_FORCE_BUILD}"

if [ ! -d "$SRC_DIR" ]; then
    echo "error: source dir '$SRC_DIR' does not exist" >&2
    exit 2
fi
if [ ! -f "$SRC_DIR/setup.py" ]; then
    echo "error: '$SRC_DIR/setup.py' not found" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/.devenv/state/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "error: devenv venv python missing at $VENV_PY (run 'devenv shell' first?)" >&2
    exit 2
fi

NV_ROOT="$REPO_ROOT/.devenv/state/venv/lib/python3.12/site-packages/nvidia"
NV_LIB_PATHS="$(find "$NV_ROOT" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')"
NV_INCLUDE_PATHS="$(find "$NV_ROOT" -maxdepth 3 -type d -name include 2>/dev/null | tr '\n' ':')"

# RTX 4090 is sm_89. Explicit list keeps build time tractable (single arch
# instead of the default "everything from sm_50 up"). Override by exporting
# TORCH_CUDA_ARCH_LIST before calling this script.
: "${TORCH_CUDA_ARCH_LIST:=8.9}"
# MAX_JOBS=4 is the proven-safe default. Each cicc instance uses 2-3 GB
# RSS at peak; 4 workers keep the compiler memory budget under 12 GB,
# which is safe on a box without swap. Bumping to 8+ is tempting on a
# 64-core host, but an incident on 2026-04-18 proved it isn't: with
# NVCC_THREADS=2 + MAX_JOBS=16 on a swap=0 system, a combination of
# leaked D-state cicc processes from a killed prior build plus 32 new
# cicc children (85 GB cicc RSS alone) produced an OOM-induced kernel
# deadlock requiring a cold reboot. Do not raise these defaults unless
# you've added swap, confirmed no leftover cicc, and have active
# monitoring on memory pressure.
: "${MAX_JOBS:=4}"
# NVCC_THREADS=1 serialises nvcc's internal threading. Works around a
# nvcc 12.4 bug where --threads 4 + complex template instantiations
# (causal-conv1d_bwd, flash-attn fwd/bwd variants) silently exits 255
# during PTX generation. Also caps per-file memory spikes.
: "${NVCC_THREADS:=1}"

# Refuse to start if stray cicc / ptxas processes survive from an
# earlier killed build — they sit in D-state, can't be reaped, and
# co-exist with new cicc children until RAM runs out.
#
# NOTE: pgrep -c prints the count AND exits 1 when zero match, so we
# swallow the non-zero status with ``|| true`` (NOT ``|| echo 0``,
# which would duplicate the count line).
_stray_cicc=$(pgrep -c cicc 2>/dev/null || true)
_stray_ptxas=$(pgrep -c ptxas 2>/dev/null || true)
_stray_cicc=${_stray_cicc:-0}
_stray_ptxas=${_stray_ptxas:-0}
if [ "$_stray_cicc" -ne 0 ] || [ "$_stray_ptxas" -ne 0 ]; then
    echo "ERROR: refusing to start — ${_stray_cicc} cicc / ${_stray_ptxas} ptxas processes still running" >&2
    echo "       (these usually indicate a prior build was killed without reaping all CUDA compiler children)" >&2
    echo "       Run: ps -C cicc -C ptxas" >&2
    echo "       Wait for them to exit, or reboot if they are wedged." >&2
    exit 3
fi

echo "=== build-cuda-ext: $SRC_DIR ==="
echo "  force-build var: $FORCE_VAR=TRUE"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  venv python:     $VENV_PY"
echo "  nvidia libs:     $(echo "$NV_LIB_PATHS" | tr ':' '\n' | wc -l) dirs"
echo

cd "$SRC_DIR"
rm -rf build dist *.egg-info

env -i \
    HOME="$HOME" \
    USER="${USER:-rch}" \
    TMPDIR="${TMPDIR:-/tmp}" \
    LANG="${LANG:-C.UTF-8}" \
    LC_ALL="${LC_ALL:-C.UTF-8}" \
    PATH="$REPO_ROOT/.devenv/state/venv/bin:/usr/local/cuda-12.4/bin:/usr/bin:/bin" \
    CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    CUDA_HOME=/usr/local/cuda-12.4 \
    CPATH="$NV_INCLUDE_PATHS:/usr/local/cuda-12.4/include" \
    LD_LIBRARY_PATH="$NV_LIB_PATHS:/usr/local/cuda-12.4/lib64" \
    LIBRARY_PATH="$NV_LIB_PATHS:/usr/local/cuda-12.4/lib64" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    NVCC_THREADS="$NVCC_THREADS" \
    "$FORCE_VAR=TRUE" \
    "$VENV_PY" setup.py bdist_wheel

echo
echo "=== wheel(s) produced ==="
ls -la "$SRC_DIR/dist/"
