#!/bin/bash
#SBATCH --job-name=raven
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python raven/test.py \
    data.modality=video \
    data/dataset=lrs3 \
    experiment_name=vsr_prelrs3vox2avs_large_ftlrs3_braven_test \
    model/visual_backbone=resnet_transformer_large \
    model.pretrained_model_path=ckpts/vsr_prelrs3vox2avs_large_ftlrs3_braven.pth \