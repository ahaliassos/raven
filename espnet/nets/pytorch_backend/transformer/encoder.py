#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
    MaskEmbedding,
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_attn_layer_type: encoder attention layer type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        frontend=None,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        relu_type="prelu",
        a_upsample_ratio=1,
        layerscale=False,
        init_values=0.1,
        ff_bn_pre=False,
        post_norm=True,
        gamma_zero=False,
        gamma_init=0.1,
        mask_init_type=None,
        mask_std_init=0.02,
        last_linear=False,
        last_norm=True,
        odim=None,
        drop_path=0.0,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding
        # -- frontend module.
        if frontend == "conv1d":
            self.frontend = Conv1dResNet(
                relu_type=relu_type,
                a_upsample_ratio=a_upsample_ratio,
                gamma_zero=gamma_zero,
                gamma_init=gamma_init,
            )
        elif frontend == "conv3d":
            self.frontend = Conv3dResNet(
                relu_type=relu_type, gamma_zero=gamma_zero, gamma_init=gamma_init
            )
        else:
            self.frontend = None
        # -- backend module.
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "vanilla_linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "token_embed":
            self.embed = MaskEmbedding(
                idim,
                attention_dim,
                pos_enc_class(attention_dim, positional_dropout_rate),
                init_type=mask_init_type,
                std_init=mask_std_init,
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if encoder_attn_layer_type == "mha":
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "legacy_rel_mha":
            encoder_attn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
                layerscale,
                init_values,
                ff_bn_pre,
                post_norm,
                drop_path,
            ),
        )
        self.after_norm = None
        if self.normalize_before and last_norm:
            self.after_norm = LayerNorm(attention_dim)

        self.last_linear = torch.nn.Linear(attention_dim, odim) if last_linear else None

    def forward(
        self, xs, masks, token_mask=None, extract_resnet_feats=False, return_feats=None
    ):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)
        if extract_resnet_feats:
            return xs

        if isinstance(self.embed, MaskEmbedding):
            xs = self.embed(xs, token_mask)
        else:
            xs = self.embed(xs)

        if return_feats:
            feats = []
            for i, e in enumerate(self.encoders):
                xs, masks = e(xs, masks)
                if return_feats == "all" or i == len(self.encoders) - 1:
                    if isinstance(xs, tuple):
                        feat = xs[0]
                    else:
                        feat = xs
                    feats.append(feat)
            return torch.stack(feats)

        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.after_norm:
            xs = self.after_norm(xs)

        if self.last_linear:
            xs = self.last_linear(xs)

        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)

        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.after_norm:
            xs = self.after_norm(xs)

        if self.last_linear:
            xs = self.last_linear(xs)

        return xs, masks, new_cache
