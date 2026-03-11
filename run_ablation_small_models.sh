#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Ablation: run baseline TCN with progressively smaller models (halve params).
#
#  Baseline: bash run_tcn_baseline_160_original_prenorm.sh \
#    --prebuilt-mmap-dir subseqs_original_mmap --lr 0.0005
#  (levels=4, nhid=80 → ~877k params)
#
#  This script runs the same command with --levels L --nhid H [--kernel-size K]
#  for each row produced by:  python ablation_model_sizes.py --list
#  (output: L  H  params  K). When K != 15, --kernel-size K is passed to get
#  models below ~1000 params (e.g. L1 H1 K5, L1 H1 K3).
#
#  Usage:
#    bash run_ablation_small_models.sh
#    bash run_ablation_small_models.sh --epochs 100   # pass extra args to training
#
#  Checkpoints go to separate dirs: checkpoints_tcn_ddp_original/<timestamp>_L4_H80/, etc.
#  Or set CHECKPOINT_SUFFIX to group runs.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS="${NGPUS:-4}"
# Optional: suffix for checkpoint dir so ablation runs don't overwrite
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-ablation}"
# Optional: run only one config (1-based index), e.g. RUN_ABLATION_INDEX=3 for SLURM array
RUN_ABLATION_INDEX="${RUN_ABLATION_INDEX:-}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

# Get ablation configs: "levels  nhid  params  kernel" per line
CONFIGS="$(python "${SCRIPT_DIR}/ablation_model_sizes.py" --list)"
if [ -z "$CONFIGS" ]; then
  echo "ablation_model_sizes.py --list returned nothing"
  exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Ablation: small models (baseline → ~1000 params, incl. small kernel)"
echo "  GPUs: ${NGPUS}  |  Extra args: $*"
echo "  Configs:"
echo "$CONFIGS" | while read -r L H _ K; do
  if [ -n "${K:-}" ] && [ "$K" != "15" ]; then echo "    levels=$L nhid=$H kernel=$K"; else echo "    levels=$L nhid=$H"; fi
done
echo "════════════════════════════════════════════════════════════════"

RUN_INDEX=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  RUN_INDEX=$((RUN_INDEX + 1))
  if [ -n "${RUN_ABLATION_INDEX}" ] && [ "$RUN_INDEX" -ne "${RUN_ABLATION_INDEX}" ]; then
    continue
  fi
  L=$(echo "$line" | awk '{print $1}')
  H=$(echo "$line" | awk '{print $2}')
  K=$(echo "$line" | awk '{print $4}')
  # Default kernel 15 (training script default)
  if [ -z "$K" ]; then K=15; fi
  CKPT_DIR="checkpoints_tcn_ddp_original/${CHECKPOINT_SUFFIX}_L${L}_H${H}"
  if [ "$K" != "15" ]; then
    CKPT_DIR="${CKPT_DIR}_K${K}"
  fi
  KERNEL_ARGS=()
  if [ "$K" != "15" ]; then
    KERNEL_ARGS=(--kernel-size "$K")
  fi
  echo ""
  echo "────────────────── Ablation run $RUN_INDEX: levels=$L nhid=$H kernel=$K ──────────────────"
  torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train_tcn_ddp_original.py" \
    --flattop-only \
    --use-prenorm \
    --lr-schedule cosine_warmup \
    --warmup-epochs 5 \
    --batch-size 16 \
    --lr 0.0005 \
    --min-lr 0.00001 \
    --levels "$L" \
    --nhid "$H" \
    "${KERNEL_ARGS[@]}" \
    --checkpoint-dir "$CKPT_DIR" \
    --no-checkpoint-by-time \
    --prebuilt-mmap-dir subseqs_original_mmap \
    "$@"
done <<< "$CONFIGS"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ablation complete."
echo "════════════════════════════════════════════════════════════════"
