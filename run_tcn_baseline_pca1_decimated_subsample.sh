#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  TCN baseline: 1D PCA decimated data (no memmap) — shape (1, 7813).
#
#  Data: decimated H5 at 100k in dsrpt_decimated_pca1; decimate 10x (--decimate-extra 10) -> 10k, 7813 samples.
#  Norm stats: norm_stats_pca1.npz (in soen_fusion_zero on remote).
#  Model: same as run_tcn_baseline_160_original_instancenorm.sh (1 input ch, InstanceNorm); dilation and layer profile unchanged; sequence length 7813.
#
#  Usage (from soen_fusion_zero on remote):
#      bash run_tcn_baseline_pca1_decimated_subsample.sh
#      bash run_tcn_baseline_pca1_decimated_subsample.sh --batch-size 16
#  Override data/norm paths:
#      DECIMATED_ROOT=/path/to/dsrpt_decimated_pca1 CLEAR_ROOT=/path/to/clear_decimated_pca1 NORM_STATS=./norm_stats_pca1.npz bash run_tcn_baseline_pca1_decimated_subsample.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

# Decimated PCA1 dirs (no memmap)
DECIMATED_ROOT="${DECIMATED_ROOT:-/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt_decimated_pca1}"
CLEAR_ROOT="${CLEAR_ROOT:-/home/idies/workspace/Storage/yhuang2/persistent/ecei/clear_decimated_pca1}"
# Norm stats: in soen_fusion_zero on remote
NORM_STATS="${NORM_STATS:-${SCRIPT_DIR}/norm_stats_pca1.npz}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  TCN — PCA1 decimated (1, 7813) + instance norm (no memmap)"
echo "  GPUs: ${NGPUS}  |  decimated: ${DECIMATED_ROOT}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

# No --prebuilt-mmap-dir: use EceiDatasetOriginal with decimated_root.
# Data at 100k; decimate 10x further (--decimate-extra 10) -> 10k, 7813 samples/window.
# nsub 781300 + data-step 10 => 78130 in file (100k); read every 10th => 7813. input-channels 1.
EXTRA=()
[[ -n "${CLEAR_ROOT:-}" ]] && EXTRA+=(--clear-decimated-root "${CLEAR_ROOT}")

# Same model as run_tcn_baseline_160_original_instancenorm.sh: levels, nhid, kernel, dilation, nrecept; same dilation/layer profile (no short-sequence cap)
torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-instance-norm \
    --clip 0.3 \
    --no-short-sequence-cap \
    --levels 4 \
    --nhid 80 \
    --kernel-size 15 \
    --dilation-base 10 \
    --nrecept-target 30000 \
    --root /home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt \
    --decimated-root "${DECIMATED_ROOT}" \
    --norm-stats "${NORM_STATS}" \
    --input-channels 1 \
    --nsub 781300 \
    --data-step 10 \
    --decimate-extra 10 \
    "${EXTRA[@]}" \
    "$@"
