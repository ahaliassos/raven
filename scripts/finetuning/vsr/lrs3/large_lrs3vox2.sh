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

srun python raven/finetune.py \
    data.modality=video \
    optimizer.lr=2e-3 \
    optimizer.lr_other=4e-3 \
    optimizer.weight_decay=0.04 \
    model.visual_backbone.drop_path=0.3 \
    optimizer.lr_decay_rate=0.75 \
    checkpoint.dirpath=/checkpoints/andreaszinonos/BYOLAV/ \
    data/dataset=lrs3 \
    experiment_name=vsr_lrs3_large_lrs3vox2 \
    model/visual_backbone=resnet_transformer_large \
    optimizer.warmup_epochs=20 \
    trainer.max_epochs=75 \
    model.pretrained_model_path=raven_large_lrs3vox2_video.pth \
    model.transfer_only_encoder=True \
    data.frames_per_gpu=2000 \
    trainer.num_nodes=4 \