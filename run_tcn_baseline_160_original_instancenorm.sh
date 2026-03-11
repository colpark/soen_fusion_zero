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
#
#  Norm stats: default idies shared path; override with NORM_STATS=/path/to/norm_stats.npz
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"
NORM_STATS="${NORM_STATS:-/home/idies/workspace/Storage/yhuang2/persistent/ecei/norm_stats.npz}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN — original dataset + instance norm (160 ch)"
echo "  GPUs: ${NGPUS}  |  norm_stats: ${NORM_STATS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-instance-norm \
    --norm-stats "${NORM_STATS}" \
    --clip 0.3 \
    "$@"
