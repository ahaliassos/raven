#!/bin/bash
#SBATCH --job-name=raven
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=2
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
    optimizer.lr=1e-3 \
    optimizer.lr_other=5e-3 \
    optimizer.weight_decay=0.1 \
    model.visual_backbone.drop_path=0.1 \
    optimizer.lr_decay_rate=0.5 \
    checkpoint.dirpath=/checkpoints/andreaszinonos/BYOLAV/ \
    data/dataset=lrs3_trainval \
    experiment_name=vsr_lrs3trainval_base_lrs3vox2 \
    model/visual_backbone=resnet_transformer_base \
    optimizer.warmup_epochs=20 \
    trainer.max_epochs=50 \
    model.visual_backbone.ddim=256 \
    model.visual_backbone.dheads=4 \
    model.visual_backbone.dunits=2048 \
    model.visual_backbone.dlayers=6 \
    model.pretrained_model_path=raven_base_lrs3vox2_video.pth \
    model.transfer_only_encoder=True \
    trainer.num_nodes=2 \