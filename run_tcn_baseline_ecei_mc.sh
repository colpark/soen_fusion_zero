#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN on ecei_mc prebuilt memmap (subseqs_mmap from
#  preprocessing_mmap_ecei_mc.ipynb). PreNorm + cosine annealing + warmup.
#
#  Batch balance: control neg:pos ratio per batch (dataset has ~40x more
#  negative samples). Use --batch-neg-pos-ratio to choose:
#    - 1 (default): most radical — 50/50 pos/neg per batch.
#    - 16: 16:1 neg:pos per batch.
#
#  Usage:
#      bash run_tcn_baseline_ecei_mc.sh
#      bash run_tcn_baseline_ecei_mc.sh --batch-neg-pos-ratio 16
#      bash run_tcn_baseline_ecei_mc.sh --prebuilt-mmap-dir /path/to/subseqs_mmap
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

# Default ecei_mc mmap (override with --prebuilt-mmap-dir)
ECEIMC_BASE="${ECEIMC_BASE:-/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc}"
DEFAULT_MMAP="${ECEIMC_BASE}/subseqs_mmap"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN (ecei_mc) — PreNorm + cosine_warmup"
echo "  batch=16, max_lr=0.0003, min_lr=1e-5"
echo "  Batch balance: --batch-neg-pos-ratio 1 (50/50) or 16 (16:1 neg:pos)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --prebuilt-mmap-dir "${DEFAULT_MMAP}" \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --batch-size 16 \
    --lr 0.0003 \
    --min-lr 0.00001 \
    --batch-neg-pos-ratio 1 \
    "$@"
