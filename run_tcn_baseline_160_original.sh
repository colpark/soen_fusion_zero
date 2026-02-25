#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN with original DisruptCNN-style dataset (shot lists, flattop, disrupt-only).
#
#  Same model and training as run_tcn_baseline_160.sh, but uses:
#  - EceiDatasetOriginal (shot list, segment logic, decimated H5)
#  - disrupt-only (no clear list)
#  - norm_stats.npz from project root (soen_fusion_zero/norm_stats.npz)
#
#  Usage:
#      bash run_tcn_baseline_160_original.sh
#      bash run_tcn_baseline_160_original.sh --epochs 100 --batch-size 32
#      bash run_tcn_baseline_160_original.sh --root /path/to/ecei/dsrpt --decimated-root /path/to/ecei/dsrpt_decimated
#
#  Data (defaults for SciServer):
#    --root              .../ecei/dsrpt (meta not used; shot list from --disrupt-file)
#    --decimated-root    .../ecei/dsrpt_decimated (flat {shot}.h5)
#    --disrupt-file      disruptcnn/shots/d3d_disrupt_ecei.final.txt
#    --flattop-only      use flattop segment only
#  Norm stats: norm_stats.npz in project root (or --norm-stats path).
#
#  Single GPU:
#    torchrun --standalone --nproc_per_node=1 train_tcn_ddp_original.py
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN — original dataset (disrupt-only, flattop, 160 ch)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    "$@"
