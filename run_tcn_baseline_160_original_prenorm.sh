#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN with original dataset + PreNorm + cosine annealing + warmup.
#
#  - PreNorm: InstanceNorm1d before each conv (no weight_norm).
#  - LR: max 0.0005, cosine annealing with 5-epoch warmup, min LR 1e-6.
#
#  Usage:
#      bash run_tcn_baseline_160_original_prenorm.sh
#      bash run_tcn_baseline_160_original_prenorm.sh --epochs 300 --batch-size 32
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN — PreNorm + cosine_warmup (max_lr=0.0005, 5ep warmup)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --lr 0.0005 \
    --min-lr 0.000001 \
    "$@"
