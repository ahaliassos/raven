# RAVEn: A PyTorch Lightning Implementation
![Overview](overview.png)
## Introduction
We provide code for the reproduction of the main results in [Jointly Learning Visual and Auditory Speech Representations from Raw Data](https://arxiv.org/abs/2212.06246). Our implementation is based on 
[PyTorch Lightning](https://www.pytorchlightning.ai/). 

## Preparation
### Installation
`conda env create -f environment.yml`. Change the environment prefix to match the location of miniconda3, if necessary.
### Data
1. The datasets used in the paper can be downloaded from the following links:
    * [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)
    * [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
    * [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
2. Compute 68 landmarks per frame using e.g., [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) and 
    [2-D FAN](https://github.com/1adrianb/face-alignment), or download them e.g., from [this repo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/models/README.md). Each landmark file should have the same name as its corresponding video (except that it ends in .npy).
3. Use the following command to crop the mouths:
    ```
    python preprocessing/extract_mouths.py --src_dir ${SOURCE_DIR} --tgt_dir ${TARGET_DIR} --landmarks_dir ${LANDMARKS_DIR}
    ``` 

## Pre-training
* We provide example sbatch scripts for pre-training in *scripts/pretraining*. For example,
    ```
    sbatch scripts/pretraining/base_lrs3.sh
    ``` 
performs pre-training with the RAVEn Base model on LRS3. It uses 4 nodes with 8 gpus per node. Tweak the scripts according to the computing environment. 
* Note: We use [hydra](https://hydra.cc/docs/intro/) for configuration management and [Weights and Biases](https://wandb.ai/site) for logging.

| Model | Pre-training dataset | Checkpoint |
|:-----:|:--------------------:|:----------:|
|  Base |         LRS3         |            |
|  Base |     LRS3+Vox2-en     |            |
| Large |     LRS3+Vox2-en     |            |

## Fine-tuning
* We provide example sbatch scripts for fine-tuning in *scripts/finetuning*. For example,
    ```
    sbatch scripts/finetuning/vsr/lrs3_trainval/large_lrs3vox2.sh
    ``` 
performs fine-tuning on LRS3-trainval with the RAVEn Large model pre-trained on LRS3+Vox2-en. It uses 2 nodes with 8 gpus per node. Tweak the scripts according to the computing environment. 

|          Model         | Pre-training dataset | WER (%) |                                           Checkpoint                                           |
|:----------------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|
|          Base          |         LRS3         |   47.0  | [Download](https://drive.google.com/file/d/1xs3VOxdqRIuBlQLmCZxsf02Qd0PGMXWf/view?usp=sharing) |
|          Base          |     LRS3+Vox2-en     |   40.2  | [Download](https://drive.google.com/file/d/14gCElgSyIa94XA0Pkc_dvZtZyz3_9kgb/view?usp=sharing) |
|          Large         |     LRS3+Vox2-en     |   32.5  | [Download](https://drive.google.com/file/d/1ueLRKgcGwSt0rJlajlMQ2QN_TctYMTUR/view?usp=sharing) |
| Large w/ self-training |     LRS3+Vox2-en     |   24.8  | [Download](https://drive.google.com/file/d/1MXQGVmSM0GeHQA5iy9Y-CJFqtJzS0wQu/view?usp=sharing) |

|          Model         | Pre-training dataset | WER (%) |                                           Checkpoint                                           |
|:----------------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|
|          Base          |         LRS3         |   4.7   | [Download](https://drive.google.com/file/d/1UJXGo9qUZ0VxPNlJfL2-JD5_428JqxNv/view?usp=sharing) |
|          Base          |     LRS3+Vox2-en     |   3.8   | [Download](https://drive.google.com/file/d/124Xu_0hB8qV2RKTbEZ5AwgckXIId4vqW/view?usp=sharing) |
|          Large         |     LRS3+Vox2-en     |   2.7   | [Download](https://drive.google.com/file/d/1vYgeo67o43XM-S24RgMJle1Acbev5fgt/view?usp=sharing) |
| Large w/ self-training |     LRS3+Vox2-en     |   2.3   | [Download](https://drive.google.com/file/d/1uCycl-je52KuLuEdMnerJYttWtORDGJ-/view?usp=sharing) |

|          Model         | Pre-training dataset | WER (%) |                                           Checkpoint                                           |
|:----------------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|
|          Base          |         LRS3         |   39.1  | [Download](https://drive.google.com/file/d/18uqnWgtVfqIFHCvEp0k6mGOWf7O6dDge/view?usp=sharing) |
|          Base          |     LRS3+Vox2-en     |   33.1  | [Download](https://drive.google.com/file/d/1qc2U5ah1NFaO94caRnsA4kOEHTJ3_P99/view?usp=sharing) |
|          Large         |     LRS3+Vox2-en     |   27.8  | [Download](https://drive.google.com/file/d/1OQZWjDYjQoApjrF3s2INsQSiS4XPXUu_/view?usp=sharing) |
| Large w/ self-training |     LRS3+Vox2-en     |   24.4  | [Download](https://drive.google.com/file/d/1tNZn_BvAVdoIIv6G14_9PvpQtsp6XSOt/view?usp=sharing) |

|          Model         | Pre-training dataset | WER (%) |                                           Checkpoint                                           |
|:----------------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|
|          Base          |         LRS3         |   2.2   | [Download](https://drive.google.com/file/d/1_vyPBj0_cepe467IdtFM1H-WCFaapdBm/view?usp=sharing) |
|          Base          |     LRS3+Vox2-en     |   1.9   | [Download](https://drive.google.com/file/d/1qcuGwTQhOttu6z8b6Rg0GwjVW4lJCthN/view?usp=sharing) |
|          Large         |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1vqUAhnR_4riYWlVOMGX5XDpGvMpzW_pe/view?usp=sharing) |
| Large w/ self-training |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1E-IPTZDX4I_YZuYbgSh4L4E7tJrUuQE8/view?usp=sharing) |

## Citation
If you find this repo useful for your research, please consider citing the following:
```bibtex
@article{haliassos2022jointly,
  title={Jointly Learning Visual and Auditory Speech Representations from Raw Data},
  author={Haliassos, Alexandros and Ma, Pingchuan and Mira, Rodrigo and Petridis, Stavros and Pantic, Maja},
  journal={arXiv preprint arXiv:2212.06246},
  year={2022}
}
```

python test.py data.modality=video experiment_name=test decode.lm_weight=0.4 model/visual_backbone=resnet_transformer_24layers model.pretrained_model_path=/datasets01/behavioural_computing_data/rodrigo/results/facemask/trainval_lrs3vox2ft_video_vox2byolavselfaud_transf24layers_warmup20_lr2em3lrother2em3_wd1em2_75ep_lrdecay0p75_droppath0p2_reinitafternorm_minlr1em5_1800frames_newlarge/model_avg_10.pth model.transfer_only_encoder=False model.pretrained_lm_path=/fsx/pingchuanma/LM_subword_data/train_transformerlm_pytorch_lm_transformer_large_ngpu8_unigram1000/rnnlm.model.best