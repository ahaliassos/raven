from fairseq.data.data_utils import compute_mask_indices
from hydra.utils import instantiate
import torch
from pytorch_lightning import LightningModule

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from schedulers.warmup_cosine import WarmupCosineScheduler


class Raven(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = instantiate(cfg.model.obj, cfg)
        self.padding_mask = self.z_vt = self.z_at = None

    def training_step(self, data, batch_idx, optimizer_idx):
        label = data["label"]
        lengths = torch.tensor(data["video_lengths"], device=data["video"].device)

        B, C, T, H, W = data["video"].shape
        mask = ~compute_mask_indices(
            (B, T),
            ~self.padding_mask,
            self.cfg.data.mask_prob,
            self.cfg.data.mask_length,
            min_masks=1,
        )
        mask = torch.from_numpy(mask).to(data["video"].device)
        mask_loss = ~mask if self.cfg.model.masked_only_loss else self.padding_mask

        if optimizer_idx == 0:
            mask_video = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            p_v_self, p_v_other, loss_ctc_v, loss_att_v, acc_v = self.model.model_video(
                (data["video"] * mask_video).squeeze(1),
                self.padding_mask.unsqueeze(-2),
                lengths,
                ~mask,
                label,
            )

            # Video-to-video loss (defaults to 0)
            loss_v2v = (
                self.model.compute_loss(p_v_self, self.z_vt, ~mask)
                if self.cfg.model.v2v_weight
                else 0
            )
            # Video-to-audio loss
            loss_v2a = (
                self.model.compute_loss(p_v_other, self.z_at, mask_loss)
                if self.cfg.model.v2a_weight
                else 0
            )

            # Log metrics for video student
            self.log(
                "acc_video_train",
                acc_v,
                on_step=False,
                on_epoch=True,
                batch_size=len(label),
            )
            self.log(
                "loss_ctc_v",
                loss_ctc_v,
                on_step=False,
                on_epoch=True,
                batch_size=len(label),
            )
            self.log(
                "loss_v2v",
                loss_v2v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(label),
            )
            self.log(
                "loss_v2a",
                loss_v2a,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(label),
            )

            return (
                self.cfg.model.v2v_weight * loss_v2v
                + self.cfg.model.v2a_weight * loss_v2a
                + loss_ctc_v
                + loss_att_v
            )

        if optimizer_idx == 1:
            mask_audio = torch.repeat_interleave(
                mask, data["audio"].size(2) // data["video"].size(2), -1
            )

            p_a_self, p_a_other, loss_ctc_a, loss_att_a, acc_a = self.model.model_audio(
                (data["audio"] * mask_audio.unsqueeze(1)).transpose(1, 2),
                self.padding_mask.unsqueeze(-2),
                lengths,
                ~mask,
                label,
            )

            # Audio-to-audio loss
            loss_a2a = (
                self.model.compute_loss(p_a_self, self.z_at, ~mask)
                if self.cfg.model.a2a_weight
                else 0.0
            )
            # Audio-to-video loss
            loss_a2v = (
                self.model.compute_loss(p_a_other, self.z_vt, mask_loss)
                if self.cfg.model.a2v_weight
                else 0.0
            )

            # Log metrics for audio student
            self.log(
                "acc_audio_train",
                acc_a,
                on_step=False,
                on_epoch=True,
                batch_size=len(label),
            )
            self.log(
                "loss_ctc_a",
                loss_ctc_a,
                on_step=False,
                on_epoch=True,
                batch_size=len(label),
            )
            self.log(
                "loss_a2a",
                loss_a2a,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(label),
            )
            self.log(
                "loss_a2v",
                loss_a2v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(label),
            )

            return (
                self.cfg.model.a2a_weight * loss_a2a
                + self.cfg.model.a2v_weight * loss_a2v
                + loss_ctc_a
                + loss_att_a
            )

    def on_train_batch_start(self, data, batch_idx):
        # Compute the targets using the video / audio teachers
        video, audio = data["video"], data["audio"]
        self.padding_mask = make_non_pad_mask(data["video_lengths"]).to(video.device)

        # Dummy mask of all zeros (i.e., no mask applied)
        mask = torch.zeros(
            (video.size(0), video.size(2)), dtype=torch.bool, device=video.device
        )
        self.z_vt, self.z_at = self.model.get_targets(
            video.squeeze(1),
            audio.transpose(1, 2),
            self.padding_mask,
            mask,
        )

    def on_train_batch_end(self, *args):
        # Update teachers via EMA and log momentum values
        momentum_video = self.momentum_scheduler_video.get_lr(self.trainer.global_step)
        self.model.update_moving_average_video(momentum_video)
        self.log(
            "momentum_video", momentum_video, on_step=False, on_epoch=True, batch_size=1
        )

        momentum_audio = self.momentum_scheduler_audio.get_lr(self.trainer.global_step)
        self.model.update_moving_average_audio(momentum_audio)
        self.log(
            "momentum_audio", momentum_audio, on_step=False, on_epoch=True, batch_size=1
        )

    def shared_val_test_step(self, data, suffix="val"):
        video, audio, label, lengths = (
            data["video"],
            data["audio"],
            data["label"],
            data["video_lengths"],
        )
        lengths = torch.tensor(lengths, device=video.device)
        padding_mask = make_non_pad_mask(lengths).to(video.device)

        mask = torch.zeros(
            (video.size(0), video.size(2)), dtype=torch.bool, device=video.device
        )

        _, _, loss_ctc_val_v, _, acc_video = self.model.model_video.backbone(
            video.squeeze(1), padding_mask.unsqueeze(-2), lengths, mask, label=label
        )
        _, _, loss_ctc_val_a, _, acc_audio = self.model.model_audio.backbone(
            audio.transpose(1, 2),
            padding_mask.unsqueeze(-2),
            lengths,
            mask,
            label=label,
        )

        self.log(f"loss_ctc_v_{suffix}", loss_ctc_val_v, batch_size=len(label))
        self.log(f"loss_ctc_a_{suffix}", loss_ctc_val_a, batch_size=len(label))
        self.log(f"acc_video_{suffix}", acc_video, batch_size=len(label))
        self.log(f"acc_audio_{suffix}", acc_audio, batch_size=len(label))

    def validation_step(self, data, batch_idx):
        self.shared_val_test_step(data)

    def test_step(self, data, batch_idx):
        self.shared_val_test_step(data, suffix="test")

    def on_train_epoch_start(self):
        # Technicality to ensure that batch sampler reloads properly each epoch
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def configure_optimizers(self):
        optimizer_video = instantiate(
            self.cfg.optimizer.optim.obj,
            self.model.model_video.parameters(),
            lr=self.cfg.optimizer.base_lr_video,
            weight_decay=self.cfg.optimizer.optim.weight_decay,
        )
        optimizer_audio = instantiate(
            self.cfg.optimizer.optim.obj,
            self.model.model_audio.parameters(),
            lr=self.cfg.optimizer.base_lr_audio,
            weight_decay=self.cfg.optimizer.optim.weight_decay,
        )

        warmup_epochs = self.cfg.optimizer.warmup_epochs
        train_len = len(self.trainer.datamodule.train_dataloader())
        scheduler_video = {
            "scheduler": WarmupCosineScheduler(
                optimizer_video,
                warmup_epochs,
                self.cfg.trainer.max_epochs,
                train_len,
            ),
            "interval": "step",
            "frequency": 1,
        }
        scheduler_audio = {
            "scheduler": WarmupCosineScheduler(
                optimizer_audio,
                warmup_epochs,
                self.cfg.trainer.max_epochs,
                train_len,
            ),
            "interval": "step",
            "frequency": 1,
        }

        self.momentum_scheduler_video = instantiate(
            self.cfg.model.momentum_scheduler,
            iter_per_epoch=train_len,
        )
        self.momentum_scheduler_audio = instantiate(
            self.cfg.model.momentum_scheduler,
            iter_per_epoch=train_len,
        )

        return [optimizer_video, optimizer_audio], [scheduler_video, scheduler_audio]
