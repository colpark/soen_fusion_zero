#!/bin/bash
# TCN ~1k-param ablation: L=1, H=1, kernel_size=5. Saves to checkpoints_tcn_ddp_original/ablation_L1_H1_k5.
# Run from repo root: sbatch disruptcnn/run_tcn_ablation_1k_instancenorm.sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --mem=200gb
#SBATCH -t 72:00:00

echo "$0"
printf "%s" "$(<$0)"
echo ""

env
echo $SLURM_JOB_NODELIST
nodes="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
echo ${nodes}

for node in ${nodes//,/ }
do
    ssh ${node} 'timeout 2400 nvidia-smi -l 1 -f '${PWD}'/nvidia.'${SLURM_JOB_ID}'.${HOSTNAME}.txt' &
done

echo "git commit"
git --git-dir=$PWD/disruptcnn/.git show --oneline -s

export CUDA_LAUNCH_BLOCKING=0

# Checkpoint folder name reflects kernel: ablation_L1_H1_k5
CKPT_DIR="checkpoints_tcn_ddp_original/ablation_L1_H1_k5"
mkdir -p "$CKPT_DIR"

file="file:///scratch/gpfs/rmc2/main_${SLURM_JOB_ID}.txt"
srun -n 16 python -u disruptcnn/main.py --dist-url $file --backend 'nccl' \
    --use-original-dataloader \
    --flattop-only \
    --use-instance-norm \
    --checkpoint-dir "$CKPT_DIR" \
    --batch-size=12 --dropout=0.1 --clip=0.3 \
    --lr=0.5 \
    --workers=6 \
    --input-channels=160 \
    --nsub 78125 \
    --epochs=1500 \
    --label-balance='const' \
    --data-step=10 --levels=1 --nrecept=30000 --nhid=1 --kernel-size=5 \
    --undersample \
    --iterations-valid 60
