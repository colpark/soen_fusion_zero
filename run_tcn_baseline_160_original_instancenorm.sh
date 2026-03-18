#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN with original dataset + InstanceNorm1d (no weight_norm).
#
#  Same as run_tcn_baseline_160_original.sh but adds --use-instance-norm so
#  the TCN uses InstanceNorm1d after each conv instead of weight normalization.
#
#  Data: ecei_mc decimated H5 — disrupt_decimated and clear_decimated.
#  Override with DECIMATED_ROOT=... CLEAR_DECIMATED_ROOT=... if needed.
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
# ecei_mc: disrupt and clear decimated H5 folders
DECIMATED_ROOT="${DECIMATED_ROOT:-/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc/disrupt_decimated}"
CLEAR_DECIMATED_ROOT="${CLEAR_DECIMATED_ROOT:-/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc/clear_decimated}"
NORM_STATS="${NORM_STATS:-/home/idies/workspace/Storage/yhuang2/persistent/ecei/norm_stats.npz}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN — original dataset + instance norm (160 ch)"
echo "  Data: disrupt=${DECIMATED_ROOT}  clear=${CLEAR_DECIMATED_ROOT}"
echo "  GPUs: ${NGPUS}  |  norm_stats: ${NORM_STATS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

EXTRA=()
[[ -n "${CLEAR_DECIMATED_ROOT:-}" ]] && EXTRA+=(--clear-decimated-root "${CLEAR_DECIMATED_ROOT}" --clear-file "${SCRIPT_DIR}/disruptcnn/shots/d3d_clear_ecei.final.txt")

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-instance-norm \
    --norm-stats "${NORM_STATS}" \
    --decimated-root "${DECIMATED_ROOT}" \
    --disrupt-file "${SCRIPT_DIR}/disruptcnn/shots/d3d_disrupt_ecei.final.txt" \
    "${EXTRA[@]}" \
    --clip 0.3 \
    "$@"
