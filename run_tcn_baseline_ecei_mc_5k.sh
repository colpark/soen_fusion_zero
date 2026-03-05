#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  TCN on 5k subsequence memmap (built from 71k via build_5k_mmap_from_71k.ipynb).
#  Uses flat layout at subseqs_mmap_5k; 50/50 batch balance (--batch-neg-pos-ratio 1).
#
#  Prerequisite: Run build_5k_mmap_from_71k.ipynb so that SUBSEQ_5K_DIR exists.
#
#  Usage:
#      bash run_tcn_baseline_ecei_mc_5k.sh
#      bash run_tcn_baseline_ecei_mc_5k.sh --prebuilt-mmap-dir /path/to/subseqs_mmap_5k
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

# 5k memmap — data path (override with SUBSEQ_5K_DIR or --prebuilt-mmap-dir)
SUBSEQ_5K_DIR="${SUBSEQ_5K_DIR:-/home/idies/workspace/Temporary/dpark1/scratch/soen_fusion_zero/subseqs_mmap_5k}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  TCN 5k (ecei_mc) — PreNorm + cosine_warmup, 50/50 batch"
echo "  Prebuilt: ${SUBSEQ_5K_DIR}"
echo "  nsub=50000 (T_sub=5k), batch=16, GPUs: ${NGPUS}"
echo "  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --prebuilt-mmap-dir "${SUBSEQ_5K_DIR}" \
    --nsub 50000 \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --batch-size 16 \
    --lr 0.0003 \
    --min-lr 0.00001 \
    --batch-neg-pos-ratio 1 \
    "$@"
