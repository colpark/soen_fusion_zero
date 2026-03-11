#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  TCN baseline: prebuilt original mmap + 1/10 time decimation + InstanceNorm.
#
#  Uses subseqs_original_mmap (flat layout: X, target, weight, train_inds, etc.)
#  with on-the-fly decimation: --decimate-factor 10 so sequence length ~7512
#  (from ~75125). Same model as instancenorm baseline; data/labels decimated
#  in the dataloader.
#
#  Usage:
#      bash run_tcn_baseline_160_original_instancenorm_subsample.sh --prebuilt-mmap-dir ./subseqs_original_mmap
#      bash run_tcn_baseline_160_original_instancenorm_subsample.sh --prebuilt-mmap-dir /path/to/subseqs_original_mmap --batch-size 16
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"
PREBUILT_MMAP_DIR="${PREBUILT_MMAP_DIR:-./subseqs_original_mmap}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  TCN — prebuilt mmap + 1/10 subsample + instance norm (160 ch)"
echo "  GPUs: ${NGPUS}  |  prebuilt: ${PREBUILT_MMAP_DIR}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-instance-norm \
    --clip 0.3 \
    --prebuilt-mmap-dir "${PREBUILT_MMAP_DIR}" \
    --decimate-factor 10 \
    "$@"
