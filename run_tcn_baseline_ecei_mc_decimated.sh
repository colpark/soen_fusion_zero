#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline TCN on ecei_mc **decimated H5** (no memmap). Reads from
#  clear_decimated/ and disrupt_decimated/ on the fly — slower than
#  run_tcn_baseline_ecei_mc.sh but no need to build subseqs_mmap first.
#
#  Same training: PreNorm, cosine_warmup, batch balance (--batch-neg-pos-ratio).
#  Requires shot lists: disrupt_file (required), clear_file (optional).
#  If --clear-file is not provided, trains disrupt-only.
#
#  Usage:
#      bash run_tcn_baseline_ecei_mc_decimated.sh
#      bash run_tcn_baseline_ecei_mc_decimated.sh --clear-file /path/to/d3d_clear_ecei.final.txt
#      bash run_tcn_baseline_ecei_mc_decimated.sh --batch-neg-pos-ratio 16
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"

ECEIMC_BASE="${ECEIMC_BASE:-/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc}"
DISRUPT_DECIMATED="${ECEIMC_BASE}/disrupt_decimated"
CLEAR_DECIMATED="${ECEIMC_BASE}/clear_decimated"
NORM_STATS="${ECEIMC_BASE}/norm_stats.npz"
# Shot lists (project-relative or absolute)
DISRUPT_FILE="${DISRUPT_FILE:-disruptcnn/shots/d3d_disrupt_ecei.final.txt}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

echo "════════════════════════════════════════════════════════════════"
echo "  Baseline TCN (ecei_mc decimated H5) — no memmap, on-the-fly load"
echo "  batch=16, PreNorm, cosine_warmup"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "════════════════════════════════════════════════════════════════"

# Do not pass --prebuilt-mmap-dir so we use EceiDatasetOriginal + decimated roots.
# Optional: add --clear-file /path/to/d3d_clear_ecei.final.txt to include clear shots.
torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --root "${ECEIMC_BASE}" \
    --decimated-root "${DISRUPT_DECIMATED}" \
    --clear-decimated-root "${CLEAR_DECIMATED}" \
    --disrupt-file "${DISRUPT_FILE}" \
    --norm-stats "${NORM_STATS}" \
    --flattop-only \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --batch-size 16 \
    --lr 0.0003 \
    --min-lr 0.00001 \
    --batch-neg-pos-ratio 1 \
    "$@"
