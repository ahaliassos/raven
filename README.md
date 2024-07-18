# (B)RAVEn: A PyTorch Lightning Implementation
## Introduction
We provide code for the reproduction of the main results in [Jointly Learning Visual and Auditory Speech Representations from Raw Data](https://arxiv.org/abs/2212.06246) and [BRAVEn: Improving Self-Supervised Pre-training for Visual and Auditory Speech Recognition
](https://arxiv.org/abs/2404.02098). Our implementation is based on 
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

## BRAVEn pre-trained models
Below are the checkpoints of the Base, Base+, and Large models pre-trained with BRAVEn.

| Model | Modality |                                           Checkpoint                                           |
|:-----:|----------|:----------------------------------------------------------------------------------------------:|
|  Base (LRS3) | Video    | [Download](https://drive.google.com/file/d/1H-kgb-mZW-3zv5D04F1wPriLnr4AXzml/view?usp=sharing) |
|  Base (LRS3) | Audio    | [Download](https://drive.google.com/file/d/1oBvymfK_eUypPofXE5pnkImv3MtJjlxR/view?usp=sharing) |
| Base+ (LRS3+Vox2) | Video    | [Download](https://drive.google.com/file/d/1dZKw4rZvzvjy92fLLQksk2RI0ubPHRB2/view?usp=sharing) |
| Base+ (LRS3+Vox2) | Audio    | [Download](https://drive.google.com/file/d/1wxGFcKwXJI7ukGZZdxEsZ6WUlI9K7rTl/view?usp=sharing) | 
| Large (LRS3+Vox2+AVS) | Video    | [Download](https://drive.google.com/file/d/1QPLgmxXUplqgS5pCcg3s7vUw-d1EzlJr/view?usp=sharing) |
| Large (LRS3+Vox2+AVS) | Audio    | [Download](https://drive.google.com/file/d/1LZOChHPP8e2RnPDMOshwJ2krpjTkDT66/view?usp=sharing) | 

## Testing
* Below are the checkpoints corresponding to Tables 1 and 2 for VSR and ASR on LRS3. Models are provided for both low- and high-resource labelled data settings. In the high-resource setting, the models are fine-tuned on the full LRS3 dataset (433 hours). In the low-resource setting, they are fine-tuned on a subset ("trainval") of LRS3 (30 hours). 

* In some cases, the models were re-trained so the WER may differ slightly from the ones shown in the paper (which are also reproduced below).

* The paths for the slurm bash scripts used for inference are shown in the table below. Note that the scripts may need to be modified according to the cluster environment. 

* The language model we used in this work can be found [here](https://drive.google.com/file/d/1mTeynSf6Sryh_mnVabpw-UwTRrh-mDc1/view?usp=sharing). 

### VSR
#### RAVEn low-resource

|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   47.0  | [Download](https://drive.google.com/file/d/1xs3VOxdqRIuBlQLmCZxsf02Qd0PGMXWf/view?usp=sharing) | scripts/vsr/lrs3_trainval/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   40.2  | [Download](https://drive.google.com/file/d/14gCElgSyIa94XA0Pkc_dvZtZyz3_9kgb/view?usp=sharing) | scripts/vsr/lrs3_trainval/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   32.5  | [Download](https://drive.google.com/file/d/1ueLRKgcGwSt0rJlajlMQ2QN_TctYMTUR/view?usp=sharing) | scripts/vsr/lrs3_trainval/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   24.8  | [Download](https://drive.google.com/file/d/1MXQGVmSM0GeHQA5iy9Y-CJFqtJzS0wQu/view?usp=sharing) | scripts/vsr/lrs3_trainval/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   23.8  |                                        same as last row                                        | scripts/vsr/lrs3_trainval/large_lrs3vox2_self_lm.sh |

#### BRAVEn low-resource

|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|    Base          |         LRS3         |   43.4  | [Download](https://drive.google.com/file/d/1ui0vFdkt6FAoDFmlGf5AgGNRL4XyXczD/view?usp=sharing)     | scripts/vsr/lrs3_trainval/base_lrs3_braven.sh              |
|    Base Plus     |     LRS3+Vox2-en     |   35.1  | [Download](https://drive.google.com/file/d/1GO-TpphYZZy4KfGzlwy-2Yw7ng98JKpx/view?usp=sharing)     | scripts/vsr/lrs3_trainval/baseplus_lrs3vox2_braven.sh      |
|    Large         |     LRS3+Vox2-en     |   30.8  | [Download](https://drive.google.com/file/d/1Cvb1VUyWFew8wggY558liAH-PRXzDfh6/view?usp=sharing)     | scripts/vsr/lrs3_trainval/large_lrs3vox2_braven.sh         |
|    Large         |     LRS3+Vox2-en+AVS |   24.8  | [Download](https://drive.google.com/file/d/1GiSlZwTcCr5ZrZdntOqubZrdcsH7shel/view?usp=sharing)     | scripts/vsr/lrs3_trainval/large_lrs3vox2avs_braven.sh      |
|    Large w/ ST   |     LRS3+Vox2-en+AVS |   21.3  | [Download](https://drive.google.com/file/d/1Udk0turivyPLqQpHc6r9H_1X01H6KoTw/view?usp=sharing)     | scripts/vsr/lrs3_trainval/large_lrs3vox2avs_self_braven.sh |
| Large w/ ST + LM |     LRS3+Vox2-en+AVS |   20.0  | same as last row | scripts/vsr/lrs3_trainval/large_lrs3vox2avs_self_lm_braven.sh |

#### RAVEn high-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|--------------------------------------------|
|       Base       |         LRS3         |   39.1  | [Download](https://drive.google.com/file/d/18uqnWgtVfqIFHCvEp0k6mGOWf7O6dDge/view?usp=sharing) | scripts/vsr/lrs3/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   33.1  | [Download](https://drive.google.com/file/d/197pP1ie8pgELIzlFNZdVqQqfpsf9o7RB/view?usp=sharing) | scripts/vsr/lrs3/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   27.8  | [Download](https://drive.google.com/file/d/1OQZWjDYjQoApjrF3s2INsQSiS4XPXUu_/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   24.4  | [Download](https://drive.google.com/file/d/1tNZn_BvAVdoIIv6G14_9PvpQtsp6XSOt/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   23.1  |                                        same as last row                                        | scripts/vsr/lrs3/large_lrs3vox2_self_lm.sh |

#### BRAVEn high-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|--------------------------------------------|
|       Base       |         LRS3         |   36.0  | [Download](https://drive.google.com/file/d/1GuAoaG9z29oI3ipRGULz_qMU8yD9NfxH/view?usp=sharing) | scripts/vsr/lrs3/base_lrs3_braven.sh              |
|       Base Plus  |     LRS3+Vox2-en     |   28.8  | [Download](https://drive.google.com/file/d/1WH6dQ4jQRlhyWKNe5y1BXITDf-PanErm/view?usp=sharing) | scripts/vsr/lrs3/baseplus_lrs3vox2_braven.sh          |
|       Large      |     LRS3+Vox2-en     |   26.6  | [Download](https://drive.google.com/file/d/1xB8UmJLp21lpuwy0zsvyTcqW2i6XVVby/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2_braven.sh         |
|       Large      |     LRS3+Vox2-en+AVS |   23.6  | [Download](https://drive.google.com/file/d/1gjHOR-zvavxAMk1vTQmkodDT11jXyeJx/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2avs_braven.sh         |
|    Large w/ ST   |     LRS3+Vox2-en+AVS |   20.9  | [Download](https://drive.google.com/file/d/1bU-bFxEiXNXNoOLKaAz6_9j7XuE7f_6r/view?usp=sharing) | scripts/vsr/lrs3/large_lrs3vox2avs_self_braven.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en+AVS |   20.1  |  same as last row   | scripts/vsr/lrs3/large_lrs3vox2avs_self_lm_braven.sh |


### ASR
#### RAVEn low-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   4.7   | [Download](https://drive.google.com/file/d/1UJXGo9qUZ0VxPNlJfL2-JD5_428JqxNv/view?usp=sharing) | scripts/asr/lrs3_trainval/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   3.8   | [Download](https://drive.google.com/file/d/124Xu_0hB8qV2RKTbEZ5AwgckXIId4vqW/view?usp=sharing) | scripts/asr/lrs3_trainval/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   2.7   | [Download](https://drive.google.com/file/d/1vYgeo67o43XM-S24RgMJle1Acbev5fgt/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   2.3   | [Download](https://drive.google.com/file/d/1uCycl-je52KuLuEdMnerJYttWtORDGJ-/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   1.9   |                                        same as last row                                        | scripts/asr/lrs3_trainval/large_lrs3vox2_self_lm.sh |

#### BRAVEn low-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   4.0   | [Download](https://drive.google.com/file/d/10EiQCFSvAip5FUpnLUUqX-Dvyu9dgmYW/view?usp=sharing) | scripts/asr/lrs3_trainval/base_lrs3_braven.sh              |
|       Base Plus  |     LRS3+Vox2-en     |   3.0   | [Download](https://drive.google.com/file/d/1yXPjsvTasSrFW4irrlrPxbsReynKp0Xg/view?usp=sharing) | scripts/asr/lrs3_trainval/baseplus_lrs3vox2_braven.sh          |
|       Large      |     LRS3+Vox2-en     |   2.3   | [Download](https://drive.google.com/file/d/1Twt9DN_CioBbhFpZ2fJg7c7mMFY8DtV_/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2_braven.sh         |
|       Large      |     LRS3+Vox2-en+AVS |   2.1   | [Download](https://drive.google.com/file/d/1CTJAWLJrb0hpfiGeUpB2G8vEn28kegRN/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2avs_braven.sh         |
|    Large w/ ST   |     LRS3+Vox2-en+AVS |   1.9   | [Download](https://drive.google.com/file/d/1Hq3oxKx-CRF7fuR0WJRBBpRGkO3f11m3/view?usp=sharing) | scripts/asr/lrs3_trainval/large_lrs3vox2avs_self_braven.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en+AVS |   1.7   | same as last row | scripts/asr/lrs3_trainval/large_lrs3vox2avs_self_lm_braven.sh |

#### RAVEn high-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   2.2   | [Download](https://drive.google.com/file/d/1_vyPBj0_cepe467IdtFM1H-WCFaapdBm/view?usp=sharing) | scripts/asr/lrs3/base_lrs3.sh              |
|       Base       |     LRS3+Vox2-en     |   1.9   | [Download](https://drive.google.com/file/d/1qcuGwTQhOttu6z8b6Rg0GwjVW4lJCthN/view?usp=sharing) | scripts/asr/lrs3/base_lrs3vox2.sh          |
|       Large      |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1vqUAhnR_4riYWlVOMGX5XDpGvMpzW_pe/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2.sh         |
|    Large w/ ST   |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1E-IPTZDX4I_YZuYbgSh4L4E7tJrUuQE8/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2_self.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en     |   1.4   |                                        same as last row                                        | scripts/asr/lrs3/large_lrs3vox2_self_lm.sh |

#### BRAVEn high-resource
|       Model      | Pre-training dataset | WER (%) |                                           Checkpoint                                           | Bash script                                         |
|:----------------:|:--------------------:|:-------:|:----------------------------------------------------------------------------------------------:|-----------------------------------------------------|
|       Base       |         LRS3         |   1.9   | [Download](https://drive.google.com/file/d/1RAwSs1FANektUbIY-AoAlRrzTGSTwGhl/view?usp=sharing) | scripts/asr/lrs3/base_lrs3_braven.sh              |
|       Base Plus  |     LRS3+Vox2-en     |   1.4   | [Download](https://drive.google.com/file/d/1vQCT9V49Qd66EuhbtfrYlbFnE7Vfj02o/view?usp=sharing) | scripts/asr/lrs3/baseplus_lrs3vox2_braven.sh          |
|       Large      |     LRS3+Vox2-en     |   1.2   | [Download](https://drive.google.com/file/d/12HTqgM09X1XwwepUrpOtgVZ1rj8ToJKm/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2_braven.sh         |
|       Large      |     LRS3+Vox2-en+AVS |   1.2   | [Download](https://drive.google.com/file/d/1J1rUm1pzsOeAWj6yVrmHCNMg_MSQzjFF/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2avs_braven.sh         |
|    Large w/ ST   |     LRS3+Vox2-en+AVS |   1.2   | [Download](https://drive.google.com/file/d/1v_OPL9ZcEGeT8cSgLO5QX16esVFAuzbq/view?usp=sharing) | scripts/asr/lrs3/large_lrs3vox2avs_self_braven.sh    |
| Large w/ ST + LM |     LRS3+Vox2-en+AVS |   1.1   | same as last row | scripts/asr/lrs3/large_lrs3vox2avs_self_lm_braven.sh |


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

```bibtex
@inproceedings{haliassos2024braven,
  title={BRAVEn: Improving Self-supervised pre-training for Visual and Auditory Speech Recognition},
  author={Haliassos, Alexandros and Zinonos, Andreas and Mira, Rodrigo and Petridis, Stavros and Pantic, Maja},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={11431--11435},
  year={2024},
  organization={IEEE}
}
```
