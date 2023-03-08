import copy

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import EMA, set_requires_grad


class BYOLSingle(nn.Module):
    def __init__(self, cfg, backbone_args=None, pred_within=None, pred_cross=None):
        super().__init__()
        self.cfg = cfg

        self.backbone = E2E(41, backbone_args)
        self.predictor_within = (
            instantiate(cfg.model.predictor) if pred_within else None
        )
        self.predictor_cross = instantiate(cfg.model.predictor) if pred_cross else None
        self.target_backbone = self.get_target_model(self.backbone.encoder)
        self.ema = EMA()
        self.target_dropout_off = cfg.model.target_dropout_off

    def update_moving_average(self, momentum):
        self.ema.update_moving_average(self.target_backbone, self.backbone, momentum)

    def get_target_model(self, model):
        target_model = copy.deepcopy(model)
        set_requires_grad(target_model, False)
        return target_model

    # Turn dropout on or off
    def set_dropout_mode(self, train_mode):
        for m in self.target_backbone.modules():
            if isinstance(m, nn.Dropout):
                if train_mode:
                    m.train()
                else:
                    m.eval()

    @torch.no_grad()
    def get_targets(self, x, padding_mask, mask):
        if self.target_dropout_off:
            self.set_dropout_mode(train_mode=False)  # Turn dropout off for teacher
        e = self.target_backbone(x, padding_mask, mask)
        if isinstance(e, tuple):
            e = e[0]
        if self.target_dropout_off:
            self.set_dropout_mode(train_mode=True)
        return e

    def get_predictions(self, x, padding_mask, lengths, mask, label):
        e_o, _, loss_ctc, loss_att, acc = self.backbone(
            x, padding_mask, lengths, mask, label, detach=True
        )

        p_within = (
            self.predictor_within(e_o, padding_mask, token_mask=mask)[0]
            if self.predictor_within
            else None
        )
        p_cross = (
            self.predictor_cross(e_o, padding_mask, token_mask=mask)[0]
            if self.predictor_cross
            else None
        )

        return p_within, p_cross, loss_ctc, loss_att, acc

    def forward(
        self, x, padding_mask, lengths, mask=None, label=None, return_targets=False
    ):
        if return_targets:
            return self.get_targets(
                x, padding_mask, mask
            )  # Create targets for other nets
        else:
            return self.get_predictions(x, padding_mask, lengths, mask, label)


class BYOLAV(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.model_video = BYOLSingle(
            cfg, cfg.model.visual_backbone, cfg.model.v2v_weight, cfg.model.v2a_weight
        )
        self.model_audio = BYOLSingle(
            cfg, cfg.model.audio_backbone, cfg.model.a2a_weight, cfg.model.a2v_weight
        )

    def update_moving_average_video(self, momentum):
        self.model_video.update_moving_average(momentum)

    def update_moving_average_audio(self, momentum):
        self.model_audio.update_moving_average(momentum)

    def get_targets(self, video, audio, padding_mask, mask):
        z_vt = self.model_video(
            video, padding_mask.unsqueeze(-2), None, mask, return_targets=True
        )
        z_at = self.model_audio(
            audio, padding_mask.unsqueeze(-2), None, mask, return_targets=True
        )
        return z_vt, z_at

    def compute_loss(self, x1, x2, mask_loss):
        loss = -F.cosine_similarity(x1, x2, dim=-1)
        loss = loss.masked_fill(mask_loss == 0, 0.0)
        return loss.sum() / mask_loss.sum()
