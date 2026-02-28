#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  PreNorm + cosine_warmup + early stopping + stronger regularization.
#
#  - PreNorm, cosine LR (max 0.0002, min 0.00002), 5-epoch warmup
#  - Batch 8/GPU (32 total), dropout 0.2, no grad clip, early stopping (patience 25)
#
#  Usage:
#      bash run_tcn_baseline_160_original_prenorm_earlystop.sh
#      bash run_tcn_baseline_160_original_prenorm_earlystop.sh --epochs 200 --prebuilt-mmap-dir subseqs_original_mmap
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  PreNorm + cosine_warmup + early stop (batch=8, lr=2e-4, no clip, dropout=0.2)"
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
    --lr 0.0002 \
    --min-lr 0.00002 \
    --batch-size 8 \
    --clip 0 \
    --dropout 0.2 \
    --early-stopping-patience 25 \
    "$@"
