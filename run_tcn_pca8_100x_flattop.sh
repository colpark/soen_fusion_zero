#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  TCN baseline: PCA-8 + 100× decimated + flattop — shape (8, 7812).
#
#  Data: single H5 from preprocessing_pca8_100x_flattop.ipynb.
#  Already pre-subsequenced, PCA-projected, and split into train/val/test.
#  Entire dataset loaded into memory.
#
#  Model: same architecture as PCA1 baseline (after decimate_extra=10
#  scaling) but with input_channels=8.
#    - kernel_size=2  (from 15, scaled by 10)
#    - dilation_base=1  (from 10, scaled by 10)
#    - nrecept_target=3000  (from 30000, scaled by 10 = 300ms at 10 kHz)
#    - levels=4, nhid=80
#    - Receptive field: 3001 samples → valid output: 4812/7812 timesteps
#
#  Usage:
#      bash run_tcn_pca8_100x_flattop.sh
#      bash run_tcn_pca8_100x_flattop.sh --epochs 100 --batch-size 32
#  Override H5 path:
#      H5_PATH=/path/to/all_data.h5 bash run_tcn_pca8_100x_flattop.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

H5_PATH="${H5_PATH:-/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc/pca8_100x_flattop/all_data.h5}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  TCN — PCA-8 + 100× decimated + flattop (8, 7812) + InstanceNorm"
echo "  GPUs: ${NGPUS}  |  H5: ${H5_PATH}"
echo "  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_pca8_h5.py" \
    --h5-path "${H5_PATH}" \
    --use-instance-norm \
    --clip 0.3 \
    --input-channels 8 \
    --kernel-size 2 \
    --dilation-base 1 \
    --nrecept-target 3000 \
    --levels 4 \
    --nhid 80 \
    "$@"
