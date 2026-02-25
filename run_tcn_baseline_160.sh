#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN: full 160 features from decimated data (no PCA)
#
#  Uses dsrpt_decimated and clear_decimated (LFS shape 20×8×T). Each
#  window is (160, T_sub) → TCN → per-timestep disruption probability.
#
#  Usage:
#      bash run_tcn_baseline_160.sh
#      bash run_tcn_baseline_160.sh --epochs 100 --batch-size 32
#      bash run_tcn_baseline_160.sh --root /path/to/ecei/dsrpt --decimated-root /path/to/ecei/dsrpt_decimated
#
#  Data (defaults for SciServer):
#    --root              .../ecei/dsrpt (meta.csv for disruptive shots)
#    --decimated-root    .../ecei/dsrpt_decimated
#    --clear-root        .../ecei/clear_decimated
#    --clear-decimated-root .../ecei/clear_decimated
#  Norm stats: norm_stats.npz (computed/saved under cwd or --norm-stats path).
#
#  Direct torchrun (same as script with 4 GPUs):
#    torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py --pca-components 0
#  Single GPU:
#    torchrun --standalone --nproc_per_node=1 train_tcn_ddp.py --pca-components 0
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN (160 channels from decimated data)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp.py" \
    --pca-components 0 \
    "$@"
