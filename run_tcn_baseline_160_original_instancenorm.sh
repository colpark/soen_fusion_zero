#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN with original dataset + InstanceNorm1d (no weight_norm).
#
#  Same as run_tcn_baseline_160_original.sh but adds --use-instance-norm so
#  the TCN uses InstanceNorm1d after each conv instead of weight normalization.
#
#  Usage:
#      bash run_tcn_baseline_160_original_instancenorm.sh
#      bash run_tcn_baseline_160_original_instancenorm.sh --epochs 100 --batch-size 32
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN — original dataset + instance norm (160 ch)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-instance-norm \
    "$@"
