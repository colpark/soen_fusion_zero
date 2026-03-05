#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  TCN on 1/10-length sequences from subseqs_mmap_all (clear + disrupt).
#  Uses same memmap but --decimate-factor 10: take every 10th time step for
#  data and labels. Receptive field 1/10 (nrecept-target 3000). Warm-cache.
#
#  Prerequisite: subseqs_mmap_all exists (from preprocessing_mmap_ecei_mc.ipynb).
#
#  Usage:
#      bash run_tcn_baseline_ecei_mc_decimated.sh
#      bash run_tcn_baseline_ecei_mc_decimated.sh --prebuilt-mmap-dir /path/to/subseqs_mmap_all
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

# Full memmap (per-split layout: train/val/test); we decimate by 10 in the loader
PREBUILT_DIR="${PREBUILT_DIR:-/home/idies/workspace/Temporary/dpark1/scratch/soen_fusion_zero/subseqs_mmap_all}"
DECIMATE=10

# Effective sequence length after 1/10 decimation (71k -> 7100 time steps)
NSUB=7100
DATA_STEP=1
# Receptive field 1/10 of full-length default (30k -> 3k)
NRECEPT_TARGET=3000

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  TCN 1/10 length (ecei_mc) — PreNorm, decimate_factor=${DECIMATE}"
echo "  Prebuilt: ${PREBUILT_DIR}"
echo "  nsub=${NSUB} (T_sub after decimation), nrecept_target=${NRECEPT_TARGET}, batch=16, GPUs: ${NGPUS}"
echo "  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --prebuilt-mmap-dir "${PREBUILT_DIR}" \
    --decimate-factor "${DECIMATE}" \
    --nsub "${NSUB}" \
    --data-step "${DATA_STEP}" \
    --nrecept-target "${NRECEPT_TARGET}" \
    --dilation-base 4 \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --batch-size 16 \
    --lr 0.0003 \
    --min-lr 0.00001 \
    --batch-neg-pos-ratio 1 \
    --num-workers 4 \
    --warm-cache \
    "$@"
