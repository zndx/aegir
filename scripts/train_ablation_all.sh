#!/usr/bin/env bash
# Train Aegir-tiny on each ablation arm sequentially, comparing
# byte-LM val loss across arms.
#
# Inputs: an ablation run directory containing per-arm chapter parquets.
# Outputs: 3 Aegir-tiny checkpoints + per-arm val loss/ppl summary.
#
# Caveats:
# - At ~200 chapters/arm (~1M tokens), the model is far from Chinchilla-
#   optimal. Differential signal between arms is expected to be small.
# - This script is for the indicative "do we see ANY signal" pass.
#   Production training waits for the 5K-chapter corpus generation.
#
# Usage::
#
#   scripts/train_ablation_all.sh /raid/checkpoints/aegir-artifacts/ablation_v1 \
#       <full_run_dir> <no_ontology_run_dir> <no_schema_run_dir>

set -euo pipefail

ABLATION_ROOT="${1:-/raid/checkpoints/aegir-artifacts/ablation_v1}"
FULL_DIR="${2:?usage: $0 ABLATION_ROOT FULL NO_ONTO NO_SCHEMA}"
NO_ONTO_DIR="${3:?usage: $0 ABLATION_ROOT FULL NO_ONTO NO_SCHEMA}"
NO_SCHEMA_DIR="${4:?usage: $0 ABLATION_ROOT FULL NO_ONTO NO_SCHEMA}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BYTES_ROOT="${ABLATION_ROOT}_bytes"
CKPT_ROOT="${ABLATION_ROOT}_ckpts"

# Hyperparameters: tiny model, short max-length (chapters fit easily), modest epochs.
MAX_LENGTH=1024
BATCH=12
EPOCHS=6
LR=3e-4

# DDP NCCL on this devenv (torch 2.9.1+cu128 / NCCL 2.27.5+cuda12.9) hits a
# 'PTX JIT compiler library not found' error during _verify_params_across_processes
# even with all nvidia/*/lib on LD_LIBRARY_PATH. Single-GPU training works
# cleanly. Aegir-tiny is 13.5M params — fits easily on one 4090.
# To diagnose & re-enable multi-GPU, set USE_DDP=1.
USE_DDP=${USE_DDP:-0}
N_PROC=${N_PROC:-1}
NV_LIBS=$(ls -d "$REPO"/.devenv/state/venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="$REPO/build/jvm-libs:$NV_LIBS:${LD_LIBRARY_PATH:-}"

mkdir -p "$BYTES_ROOT" "$CKPT_ROOT"
cd "$REPO"

convert() {
    local arm=$1
    local src_dir=$2
    local out_dir="$BYTES_ROOT/$arm"
    if [[ -f "$out_dir/train.parquet" && -f "$out_dir/val.parquet" ]]; then
        echo "=== [convert] $arm: already exists, skipping ==="
        return
    fi
    echo "=== [convert] $arm @ $(date -u) ==="
    uv run --no-sync python \
        scripts/chapters_to_byte_parquet.py \
        --chapters-parquet "$src_dir/chapters.parquet" \
        --out-dir "$out_dir" \
        --chunk-size $MAX_LENGTH \
        --val-frac 0.1 \
        --rg-size 16
}

train() {
    local arm=$1
    local out_dir="$BYTES_ROOT/$arm"
    local ckpt_dir="$CKPT_ROOT/$arm"
    local log="$ckpt_dir/train.log"
    mkdir -p "$ckpt_dir"
    echo "=== [train] $arm @ $(date -u) ==="
    if [[ "$USE_DDP" == "1" && "$N_PROC" -gt 1 ]]; then
        uv run --no-sync torchrun --nproc_per_node=$N_PROC \
            train_pretrain.py \
            --train-parquet "$out_dir/train.parquet" \
            --val-parquet "$out_dir/val.parquet" \
            --model-size tiny --max-length $MAX_LENGTH \
            --batch-size $BATCH --epochs $EPOCHS --lr $LR \
            --output-dir "$ckpt_dir" --log-interval 20 --num-workers 2 \
            2>&1 | tee "$log"
    else
        # Pin to a single GPU; do NOT collide with anything else running.
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} uv run --no-sync python \
            train_pretrain.py \
            --train-parquet "$out_dir/train.parquet" \
            --val-parquet "$out_dir/val.parquet" \
            --model-size tiny --max-length $MAX_LENGTH \
            --batch-size $BATCH --epochs $EPOCHS --lr $LR \
            --output-dir "$ckpt_dir" --log-interval 20 --num-workers 2 \
            2>&1 | tee "$log"
    fi
}

summarize() {
    echo
    echo "=== Per-arm summary ==="
    for arm in full no_ontology no_schema; do
        local log="$CKPT_ROOT/$arm/train.log"
        if [[ ! -f "$log" ]]; then
            echo "  $arm: no log"
            continue
        fi
        local best=$(grep -E "val_loss|val_ppl" "$log" | tail -3 | tr -d '\r')
        echo "  $arm:"
        echo "$best" | sed 's/^/    /'
    done
}

echo "=== ablation train pass @ $(date -u) ==="
echo "  bytes_root: $BYTES_ROOT"
echo "  ckpt_root:  $CKPT_ROOT"

convert "full"        "$FULL_DIR"
convert "no_ontology" "$NO_ONTO_DIR"
convert "no_schema"   "$NO_SCHEMA_DIR"

train "full"
train "no_ontology"
train "no_schema"

summarize

echo "=== done @ $(date -u) ==="
