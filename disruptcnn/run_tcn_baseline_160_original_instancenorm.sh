#!/bin/bash
# TCN baseline: original dataloader (shot list, flattop, Twarn=300) + InstanceNorm1d instead of weight_norm.
# Same as run_tcn_baseline_160_original.sh but with --use-instance-norm.
# Run from repo root: sbatch disruptcnn/run_tcn_baseline_160_original_instancenorm.sh
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

file="file:///scratch/gpfs/rmc2/main_${SLURM_JOB_ID}.txt"
srun -n 16 python -u disruptcnn/main.py --dist-url $file --backend 'nccl' \
    --use-original-dataloader \
    --use-instance-norm \
    --batch-size=12 --dropout=0.1 --clip=0.3 \
    --lr=0.5 \
    --workers=6 \
    --input-channels=160 \
    --nsub 78125 \
    --epochs=1500 \
    --label-balance='const' \
    --data-step=10 --levels=4 --nrecept=30000 --nhid=80 \
    --undersample \
    --iterations-valid 60
