# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging

import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if (
            self.rel_pos_type is None
            and args.transformer_encoder_attn_layer_type == "rel_mha"
        ):
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        self.encoder = Encoder(
            idim=args.idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            frontend=args.transformer_frontend,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
            layerscale=args.layerscale,
            init_values=args.init_values,
            ff_bn_pre=args.ff_bn_pre,
            post_norm=args.post_norm,
            gamma_zero=args.gamma_zero,
            gamma_init=args.gamma_init,
            mask_init_type=args.mask_init_type,
            drop_path=args.drop_path,
        )

        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.ddim,
            attention_heads=args.dheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            proj_decoder=torch.nn.Linear(args.adim, args.ddim)
            if args.adim != args.ddim
            else None,
        )

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        self.mtlalpha = args.mtlalpha

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, padding_mask, lengths, mask=None, label=None, detach=False):
        x, _ = self.encoder(x, padding_mask, mask)

        x_d = x.detach() if detach else x

        loss_ctc = self.ctc(x_d, lengths, label)

        # decoder loss (with stop-gradient)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x_d, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)

        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return x, loss, loss_ctc, loss_att, acc
