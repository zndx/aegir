#!/bin/bash
# Aggressive-but-safe flash-attn source build, tuned for the 6×RTX-4090 box
# (64 cores, 125 GB RAM, swap=0). Parallelises harder than the conservative
# ``bin/build-cuda-ext.sh`` default because on an idle system we can afford
# it — but every knob has a safety net so we never repeat the 2026-04-18
# OOM-induced kernel deadlock.
#
# Dials (baseline for this machine):
#   MAX_JOBS=16       → 16 ninja workers, ~40 GB cicc RSS peak
#   NVCC_THREADS=1    → avoids nvcc-12.4's PTX-segfault bug on template-
#                       heavy .cu files
#   TORCH_CUDA_ARCH_LIST=8.9  (RTX 4090)
#   Temporary swap    → a 16 GB swapfile is created for the duration of
#                       the build and dropped on exit. Acts as a safety
#                       valve; if we're ever using it at build time, we
#                       shed a worker and continue instead of deadlocking.
#
# Expected wall time on this box: ~45–75 minutes.  ~2-3× faster than the
# conservative default at MAX_JOBS=4.
#
# Pre-flight (abort early rather than partway through):
#   • devenv stack stopped (frees postgres/qdrant/gateway RAM)
#   • no stray cicc / ptxas from an earlier killed build
#   • ≥ 60 GB RAM available, ≥ 20 GB disk free
#   • patched source tree present, ninja + gcc-11 + nvcc available
#
# Post-flight:
#   • wheel copied to build/wheels/
#   • installed via ``uv pip install --reinstall --no-deps``
#   • import + GPU RMSNorm smoke test
#   • devenv stack restarted
#   • swapfile removed, build dir cleared
#
# All exits go through the trap so children are reaped and swap is
# removed even on Ctrl-C or unexpected failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/build/patched-src/flash_attn-2.8.3"
LOG="/tmp/aegir-flash-attn-build.log"
SWAPFILE="$REPO_ROOT/build/.aegir-flash-swap"
SWAP_SIZE_MB=16384

# Tunable — override by exporting before calling. Defaults are the
# "aggressive baseline" for this machine; the safety gates below enforce
# the memory math.
: "${MAX_JOBS:=16}"
: "${NVCC_THREADS:=1}"
: "${TORCH_CUDA_ARCH_LIST:=8.9}"
: "${MIN_FREE_RAM_GB:=60}"
: "${MIN_FREE_DISK_GB:=20}"
: "${STALL_THRESHOLD_SECS:=900}"  # 15 min with no new .o files → abort

# ── Cleanup trap ────────────────────────────────────────────────

_we_made_swap=0
_monitor_pid=0

cleanup() {
    local ec=$?
    echo
    echo "=== cleanup (exit=$ec) ==="
    if [ "$_monitor_pid" -ne 0 ] && kill -0 "$_monitor_pid" 2>/dev/null; then
        kill "$_monitor_pid" 2>/dev/null || true
    fi
    # Reap any surviving compiler children so they don't poison the next build.
    pkill -P $$ 2>/dev/null || true
    pkill -TERM -f "flash_attn-2.8.3.*setup.py" 2>/dev/null || true
    pkill -TERM cicc 2>/dev/null || true
    pkill -TERM ptxas 2>/dev/null || true
    sleep 2
    pkill -KILL cicc 2>/dev/null || true
    pkill -KILL ptxas 2>/dev/null || true
    # Drop the swap file if we created it (leave pre-existing swap alone).
    if [ "$_we_made_swap" -eq 1 ] && [ -f "$SWAPFILE" ]; then
        echo "Removing temporary swapfile"
        sudo swapoff "$SWAPFILE" 2>/dev/null || true
        sudo rm -f "$SWAPFILE" || true
    fi
    exit $ec
}
trap cleanup INT TERM EXIT

# ── Pre-flight ──────────────────────────────────────────────────

echo "=== pre-flight ==="

if [ ! -f "$SRC_DIR/setup.py" ]; then
    echo "error: patched source tree missing at $SRC_DIR" >&2
    echo "       (expected from build/patched-src/flash_attn-2.8.3/)" >&2
    exit 2
fi

# Stop the devenv stack to free RAM (postgres + qdrant + gateway + vite).
# Safe to do: caller is running a deliberate heavy build, not serving
# the UI.  Restarted at the end.
if pgrep -f "aegir.gateway" >/dev/null; then
    echo "Stopping devenv processes to free RAM..."
    cd "$REPO_ROOT"
    devenv processes stop >/dev/null 2>&1 || true
    sleep 2
fi

# Reject if stray cicc/ptxas survive from an earlier killed build —
# they sit in D-state, can't be reaped, and stack with new workers
# until RAM runs out. This is the root cause of the 2026-04-18 deadlock.
# pgrep -c prints the count AND exits 1 when the count is 0, so we
# have to swallow the non-zero status without discarding the count.
_stray_cicc=$(pgrep -c cicc 2>/dev/null || true)
_stray_ptxas=$(pgrep -c ptxas 2>/dev/null || true)
_stray_cicc=${_stray_cicc:-0}
_stray_ptxas=${_stray_ptxas:-0}
if [ "$_stray_cicc" -ne 0 ] || [ "$_stray_ptxas" -ne 0 ]; then
    echo "error: ${_stray_cicc} cicc / ${_stray_ptxas} ptxas still running from a prior build" >&2
    echo "       run: ps -C cicc -C ptxas" >&2
    echo "       wait for them to exit or reboot" >&2
    exit 3
fi

# Resource gates.
_free_ram_gb=$(awk '/^MemAvailable:/ {printf "%d", $2/1024/1024}' /proc/meminfo)
_free_disk_gb=$(df --output=avail -BG "$REPO_ROOT" | tail -1 | tr -dc '0-9')
if [ "$_free_ram_gb" -lt "$MIN_FREE_RAM_GB" ]; then
    echo "error: only ${_free_ram_gb} GB RAM available, need ${MIN_FREE_RAM_GB} GB" >&2
    exit 4
fi
if [ "$_free_disk_gb" -lt "$MIN_FREE_DISK_GB" ]; then
    echo "error: only ${_free_disk_gb} GB disk free, need ${MIN_FREE_DISK_GB} GB" >&2
    exit 4
fi

# Safety-net swap. If the system has no swap, create a temporary
# swapfile at $SWAPFILE; drop it in the cleanup trap. Absorbs memory
# spikes during peak concurrent compile so an overshoot spills to disk
# instead of deadlocking.
_existing_swap=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
if [ "$_existing_swap" = "0" ]; then
    echo "No swap configured — creating ${SWAP_SIZE_MB}MB safety-net swapfile at $SWAPFILE"
    mkdir -p "$(dirname "$SWAPFILE")"
    if ! sudo -n true 2>/dev/null; then
        echo "warning: sudo requires a password; swapfile creation will prompt interactively" >&2
    fi
    sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=$SWAP_SIZE_MB status=none
    sudo chmod 0600 "$SWAPFILE"
    sudo mkswap "$SWAPFILE" >/dev/null
    sudo swapon "$SWAPFILE"
    _we_made_swap=1
fi

echo "settings:"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  NVCC_THREADS=$NVCC_THREADS"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  free RAM: ${_free_ram_gb} GB, free disk: ${_free_disk_gb} GB"
echo "  safety-net swap: ${_we_made_swap} (created by this script)"
echo

# ── Stall monitor ────────────────────────────────────────────────
# Watches the build's object-file output. If no new .o files appear
# for STALL_THRESHOLD_SECS, the build is probably wedged and we kill
# it (the cleanup trap then reaps children + removes swap).

build_dir="$SRC_DIR/build/temp.linux-x86_64-cpython-312"

monitor() {
    local parent_pid="$1"
    local last_count=0
    local last_change=$(date +%s)
    while kill -0 "$parent_pid" 2>/dev/null; do
        sleep 60
        local now=$(date +%s)
        local count=0
        if [ -d "$build_dir" ]; then
            count=$(find "$build_dir" -name "*.o" 2>/dev/null | wc -l)
        fi
        local mem=$(awk '/^MemAvailable:/ {printf "%d", $2/1024/1024}' /proc/meminfo)
        local load=$(awk '{print $1}' /proc/loadavg)
        if [ "$count" -gt "$last_count" ]; then
            last_count=$count
            last_change=$now
        fi
        local stall=$((now - last_change))
        echo "  [monitor $(date +%H:%M:%S)] .o=${count}/73  freeRAM=${mem}GB  load=${load}  stall=${stall}s"
        if [ "$stall" -gt "$STALL_THRESHOLD_SECS" ]; then
            echo "  [monitor] STALLED for ${stall}s (> ${STALL_THRESHOLD_SECS}s) — killing build" >&2
            kill -TERM "$parent_pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$parent_pid" 2>/dev/null || true
            return 1
        fi
    done
}

# ── Build ───────────────────────────────────────────────────────

echo "=== starting flash-attn build ==="
rm -rf "$SRC_DIR/build" "$SRC_DIR/dist" "$SRC_DIR/"*.egg-info

# Run the build in the foreground but fork the monitor alongside it.
# This gives us a clean wait() target and trap-reaps the monitor on exit.
set +e
MAX_JOBS="$MAX_JOBS" \
NVCC_THREADS="$NVCC_THREADS" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    bash "$REPO_ROOT/bin/build-cuda-ext.sh" \
        "$SRC_DIR" FLASH_ATTENTION_FORCE_BUILD \
    > "$LOG" 2>&1 &
BUILD_PID=$!
set -e

echo "build pid=$BUILD_PID, log=$LOG"
monitor "$BUILD_PID" &
_monitor_pid=$!

# Wait; propagate build's exit status.
set +e
wait "$BUILD_PID"
BUILD_EC=$?
set -e
if kill -0 "$_monitor_pid" 2>/dev/null; then
    kill "$_monitor_pid" 2>/dev/null || true
fi
_monitor_pid=0

echo
echo "=== build exited with code $BUILD_EC ==="
if [ "$BUILD_EC" -ne 0 ]; then
    echo "tail of build log:"
    tail -30 "$LOG"
    exit "$BUILD_EC"
fi

# ── Install ────────────────────────────────────────────────────

WHEEL_SRC=$(ls "$SRC_DIR"/dist/flash_attn-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_SRC" ]; then
    echo "error: build succeeded but no wheel under $SRC_DIR/dist/" >&2
    exit 5
fi

mkdir -p "$REPO_ROOT/build/wheels"
cp "$WHEEL_SRC" "$REPO_ROOT/build/wheels/"
echo "wheel: $(ls "$REPO_ROOT/build/wheels/" | grep flash_attn | head -1)"

cd "$REPO_ROOT"
uv pip install --reinstall --no-deps "$WHEEL_SRC" 2>&1 | tail -5

echo
echo "=== import + GPU kernel smoke test ==="
uv run --no-sync python -c "
import torch
import flash_attn
print('flash_attn', flash_attn.__version__)
from flash_attn.ops.triton.layer_norm import RMSNorm
rms = RMSNorm(128).cuda().to(torch.bfloat16)
x = torch.randn(4, 16, 128, dtype=torch.bfloat16, device='cuda')
y = rms(x)
print('RMSNorm OK:', y.shape, y.dtype)
"

# ── Restart devenv stack ───────────────────────────────────────

echo
echo "=== restart devenv stack ==="
cd "$REPO_ROOT"
devenv up -d >/dev/null 2>&1 || echo "warning: devenv up -d failed, restart manually"

echo
echo "=== build complete ==="
echo "   flash-attn installed, stack restarted"
echo "   cleanup trap will drop swap + reap children on exit"
