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
    data.modality=audio \
    optimizer.lr=1e-3 \
    optimizer.lr_other=2e-3 \
    optimizer.weight_decay=0.04 \
    data.timemask_window=0.6 \
    data.timemask_stride=2 \
    model.audio_backbone.drop_path=0.3 \
    optimizer.lr_decay_rate=0.75 \
    checkpoint.dirpath=/checkpoints/andreaszinonos/BYOLAV/ \
    data/dataset=lrs3_trainval \
    experiment_name=asr_lrs3_large_lrs3vox2 \
    model/audio_backbone=resnet_transformer_large \
    optimizer.warmup_epochs=20 \
    trainer.max_epochs=150 \
    model.pretrained_model_path=raven_large_lrs3vox2_audio.pth \
    model.transfer_only_encoder=True \
    data.frames_per_gpu=2000 \
    trainer.num_nodes=4 \