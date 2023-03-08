#!/bin/bash
#SBATCH --job-name=raven
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue
#SBATCH --output=/fsx/andreaszinonos/Other_Repos/BYOLAV/output/slurm/slurm-%j.out

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1

srun python raven/pretrain.py \
    optimizer.base_lr_video=3e-3 \
    checkpoint.dirpath=/checkpoints/andreaszinonos/BYOLAV/ \
    data/dataset=lrs3 \
    experiment_name=raven_base_lrs3 \
    model/visual_backbone=resnet_transformer_base \
    optimizer.warmup_epochs=40 \
    trainer.max_epochs=150 \
    trainer.num_nodes=4 \