#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Distributed TCN training — 4-GPU launch script
#
#  Usage:
#      bash run_train.sh                    # defaults (AdamW, 48/GPU)
#      bash run_train.sh --lr 0.5 --optimizer sgd   # match disruptcnn
#      bash run_train.sh --batch-size 24             # smaller per GPU
#      bash run_train.sh --resume checkpoints_tcn_ddp/last.pt
#
#  The script uses torchrun (elastic launch) with NCCL backend.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── GPU config ────────────────────────────────────────────────────────
NGPUS=4                        # number of GPUs on the node

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_tcn_ddp.py"

# ── NCCL tuning (sensible defaults for single-node multi-GPU) ─────────
export NCCL_IB_DISABLE=1              # no InfiniBand on single node
export OMP_NUM_THREADS=4              # OpenMP threads per worker

# ── Launch ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  Launching distributed training on ${NGPUS} GPUs"
echo "  Script: ${TRAIN_SCRIPT}"
echo "  Extra args: $@"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${TRAIN_SCRIPT}" \
    "$@"
