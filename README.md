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

## RAVEn pre-trained models
Below are the checkpoints of the Base and Large models pre-trained with RAVEn on **LRS3+Vox2-en**.

| Model | Modality |                                           Checkpoint                                           |
|:-----:|----------|:----------------------------------------------------------------------------------------------:|
|  Base | Video    | [Download](https://drive.google.com/file/d/14Iz3l1I0NjNKT3kI4uptCk4rNkyDoU2N/view?usp=sharing) |
|  Base | Audio    | [Download](https://drive.google.com/file/d/19yoKn58g37x1o_oo-zko_dg27N8-bVIJ/view?usp=sharing) |
| Large | Video    | [Download](https://drive.google.com/file/d/1wKqUIyhkzqlfmdF1zHrEizQtX2NQfjHC/view?usp=sharing) |
| Large | Audio    | [Download](https://drive.google.com/file/d/1hBa1QTbDDjFcK8E92s-6gA4chEc2WnaE/view?usp=sharing) | 

## Testing
* Below are the checkpoints corresponding to Tables 1 and 2 for VSR and ASR on LRS3. Models are provided for both low- and high-resource labelled data settings. In the high-resource setting, the models are fine-tuned on the full LRS3 dataset (433 hours). In the low-resource setting, they are fine-tuned on a subset ("trainval") of LRS3 (30 hours). 

* In some cases, the models were re-trained so the WER may differ slightly from the ones shown in the paper (which are also reproduced below).

* The paths for the slurm bash scripts used for inference are shown in the table below. Note that the scripts may need to be modified according to the cluster environment. 

* The language model we used in this work can be found [here](https://drive.google.com/file/d/1mTeynSf6Sryh_mnVabpw-UwTRrh-mDc1/view?usp=sharing). 

### VSR
#### Low-resource

|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   47.0  | [Download](https://drive.google.com/file/d/1xs3VOxdqRIuBlQLmCZxsf02Qd0PGMXWf/view?usp=sharing) | scripts/vsr/lrs3_trainval/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   40.2  | [Download](https://drive.google.com/file/d/14gCElgSyIa94XA0Pkc_dvZtZyz3_9kgb/view?usp=sharing) | scripts/vsr/lrs3_trainval/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   32.5  | [Download](https://drive.google.com/file/d/1ueLRKgcGwSt0rJlajlMQ2QN_TctYMTUR/view?usp=sharing) | scripts/vsr/lrs3_trainval/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   24.8  | [Download](https://drive.google.com/file/d/1MXQGVmSM0GeHQA5iy9Y-CJFqtJzS0wQu/view?usp=sharing) | scripts/vsr/lrs3_trainval/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   23.8  |                                        same as last row                                        | scripts/vsr/lrs3_trainval/large_lrs3vox2_self_lm.sh |

#### High-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|--------------------------------------------|
|       Base       |         LRS3         |   39.1  | [Download](https://drive.google.com/file/d/18uqnWgtVfqIFHCvEp0k6mGOWf7O6dDge/view?usp=sharing) | scripts/vsr/lrs3/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   33.1  | [Download](https://drive.google.com/file/d/197pP1ie8pgELIzlFNZdVqQqfpsf9o7RB/view?usp=sharing) | scripts/vsr/lrs3/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   27.8  | [Download](https://drive.google.com/file/d/1OQZWjDYjQoApjrF3s2INsQSiS4XPXUu_/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   24.4  | [Download](https://drive.google.com/file/d/1tNZn_BvAVdoIIv6G14_9PvpQtsp6XSOt/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   23.1  |                                        same as last row                                        | scripts/vsr/lrs3/large_lrs3vox2_self_lm.sh |
### ASR
#### Low-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   4.7   | [Download](https://drive.google.com/file/d/1UJXGo9qUZ0VxPNlJfL2-JD5_428JqxNv/view?usp=sharing) | scripts/asr/lrs3_trainval/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   3.8   | [Download](https://drive.google.com/file/d/124Xu_0hB8qV2RKTbEZ5AwgckXIId4vqW/view?usp=sharing) | scripts/asr/lrs3_trainval/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   2.7   | [Download](https://drive.google.com/file/d/1vYgeo67o43XM-S24RgMJle1Acbev5fgt/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   2.3   | [Download](https://drive.google.com/file/d/1uCycl-je52KuLuEdMnerJYttWtORDGJ-/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   1.9   |                                        same as last row                                        | scripts/asr/lrs3_trainval/large_lrs3vox2_self_lm.sh |

#### High-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   2.2   | [Download](https://drive.google.com/file/d/1_vyPBj0_cepe467IdtFM1H-WCFaapdBm/view?usp=sharing) | scripts/asr/lrs3/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   1.9   | [Download](https://drive.google.com/file/d/1qcuGwTQhOttu6z8b6Rg0GwjVW4lJCthN/view?usp=sharing) | scripts/asr/lrs3/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1vqUAhnR_4riYWlVOMGX5XDpGvMpzW_pe/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1E-IPTZDX4I_YZuYbgSh4L4E7tJrUuQE8/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   1.4   |                                        same as last row                                        | scripts/asr/lrs3/large_lrs3vox2_self_lm.sh |

Code for pre-training and fine-tuning coming soon...

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