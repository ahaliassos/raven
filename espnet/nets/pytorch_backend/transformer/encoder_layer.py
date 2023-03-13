#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import copy
import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.norm import LayerNorm, BatchNorm1d


# Taken from https://github.com/rwightman/pytorch-image-models/blob/f670d98cb8ec70ed6e03b4be60a18faf4dc913b5/timm/models/layers/drop.py#L157
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
        RelPositionMultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.transformer.convolution.
        ConvolutionModule feed_foreard:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool macaron_style: whether to use macaron style for PositionwiseFeedForward

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
        layerscale=False,
        init_values=0.0,
        ff_bn_pre=False,
        post_norm=True,
        drop_path=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.ff_scale = 1.0
        self.conv_module = conv_module
        self.macaron_style = macaron_style
        self.norm_ff = (
            BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)
        )  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if self.macaron_style:
            self.feed_forward_macaron = copy.deepcopy(feed_forward)
            self.ff_scale = 0.5
            # for another FNN module in macaron style
            self.norm_ff_macaron = BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)
        if self.conv_module is not None:
            self.norm_conv = (
                BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)
            )  # for the CNN module
            if post_norm:
                self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.ff_bn_pre = ff_bn_pre
        self.post_norm = post_norm
        self.layerscale = layerscale
        if layerscale:
            self.gamma_ff = nn.Parameter(
                init_values * torch.ones((size,)), requires_grad=True
            )
            self.gamma_mha = nn.Parameter(
                init_values * torch.ones((size,)), requires_grad=True
            )
            if self.macaron_style:
                self.gamma_ff_macaron = nn.Parameter(
                    init_values * torch.ones((size,)), requires_grad=True
                )
            if self.conv_module is not None:
                self.gamma_conv = nn.Parameter(
                    init_values * torch.ones((size,)), requires_grad=True
                )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
                x = self.norm_ff_macaron(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
            if self.layerscale:
                x = residual + self.drop_path(
                    self.ff_scale
                    * self.dropout(self.gamma_ff_macaron * self.feed_forward_macaron(x))
                )
            else:
                x = residual + self.drop_path(
                    self.ff_scale * self.dropout(self.feed_forward_macaron(x))
                )
            if not self.normalize_before:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
                x = self.norm_ff_macaron(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            if self.layerscale:
                x = residual + self.drop_path(
                    self.gamma_mha * self.concat_linear(x_concat)
                )
            else:
                x = residual + self.drop_path(self.concat_linear(x_concat))
        else:
            if self.layerscale:
                x = residual + self.drop_path(self.dropout(self.gamma_mha * x_att))
            else:
                x = residual + self.drop_path(self.dropout(x_att))
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
                x = self.norm_conv(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
            if self.layerscale:
                x = residual + self.drop_path(
                    self.dropout(self.gamma_conv * self.conv_module(x))
                )
            else:
                x = residual + self.drop_path(self.dropout(self.conv_module(x)))
            if not self.normalize_before:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)
                x = self.norm_conv(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2)

        # feed forward module
        residual = x
        if self.normalize_before:
            if self.ff_bn_pre:
                x = x.transpose(1, 2)
            x = self.norm_ff(x)
            if self.ff_bn_pre:
                x = x.transpose(1, 2)
        if self.layerscale:
            x = residual + self.drop_path(
                self.ff_scale * self.dropout(self.gamma_ff * self.feed_forward(x))
            )
        else:
            x = residual + self.drop_path(
                self.ff_scale * self.dropout(self.feed_forward(x))
            )
        if not self.normalize_before:
            if self.ff_bn_pre:
                x = x.transpose(1, 2)
            x = self.norm_ff(x)
            if self.ff_bn_pre:
                x = x.transpose(1, 2)

        if self.conv_module is not None and self.post_norm:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask
