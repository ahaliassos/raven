import os

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from .dataset import AVDataset
from .samplers import ByFrameCountSampler, DistributedSamplerWrapper, RandomSamplerWrapper
from .transforms import AdaptiveLengthTimeMask


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) < 3:
        collated_batch = collated_batch.unsqueeze(1)
    else:
        collated_batch = collated_batch.permute((0, 4, 1, 2, 3))  # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == 'label' else 0.0
        c_batch, sample_lengths = pad([s[data_type] for s in batch if s[data_type] is not None], pad_val)
        batch_out[data_type] = c_batch
        batch_out[data_type + '_lengths'] = sample_lengths
    
    return batch_out


class DataModule(LightningDataModule):

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        print('total gpus:', self.total_gpus)

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
            Lambda(lambda x: x / 255.),
        ] + (
            [
                RandomCrop(args.crop_type.random_crop_dim),
                Resize(args.crop_type.resize_dim),
                RandomHorizontalFlip(args.horizontal_flip_prob)
            ]
            if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim), Resize(args.crop_type.resize_dim)]
        )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend([Lambda(lambda x: x.transpose(0, 1)), Grayscale(), Lambda(lambda x: x.transpose(0, 1))])
        transform.append(instantiate(args.channel.obj))

        if mode == "train" and args.use_masking:
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 25),
                    stride=int(args.timemask_stride * 25),
                    replace_with_zero=True
                )
            )

        return Compose(transform)

    def _audio_transform(self, mode):
        args = self.cfg.data
        transform = [Lambda(lambda x: x)]

        if mode == "train" and args.use_masking:
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 16000),
                    stride=int(args.timemask_stride * 16000),
                    replace_with_zero=True
                    )
                )

        return Compose(transform)

    def _dataloader(self, ds, sampler, collate_fn):
        return DataLoader(
            ds,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video = self._video_transform(mode='val')
        transform_audio = self._audio_transform(mode='val')

        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        test_ds = AVDataset(
            data_path=os.path.join(parent_path, "data_paths", ds_args.test_csv),
            video_path_prefix_lrs2=ds_args.paths.root_lrs2_video,
            audio_path_prefix_lrs2=ds_args.paths.root_lrs2_audio,
            video_path_prefix_lrs3=ds_args.paths.root_lrs3_video,
            # video_path_prefix_lrs3="/vol/paramonos2/datasets2/LRS3v04/cropped_mouths_uncompressed_gray_with_audio",
            audio_path_prefix_lrs3=ds_args.paths.root_lrs3_audio,
            # audio_path_prefix_lrs3="/vol/paramonos2/projects/pm4115/LRS3/LRS3_audio",
            transforms={'video': transform_video, 'audio': transform_audio},
            modality=self.cfg.data.modality,
        )
        sampler = ByFrameCountSampler(test_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(test_ds, sampler, collate_pad)
