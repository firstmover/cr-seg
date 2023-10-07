#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 07/21/2023
#
# Distributed under terms of the MIT license.

"""Build voxelmorph model following mmengine API
"""

import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement

from .layers import SpatialTransformer
from .losses import VOXELMORPH_LOSS, Grad
from .networks import Unet


@MODELS.register_module()
class VoxelMorph(BaseModel):
    def __init__(
        self,
        inshape: tuple = (32, 32),
        unet_in_channel: int = 2,
        unet_channel_cfg=(
            (16, 32, 32, 32),  # encoder features
            (32, 32, 32, 32, 32, 16, 16),  # decoder features
        ),
        lambda_grad: float = 0.1,
        loss_cfg: tp.Optional[dict] = None,
        **kwargs
    ):
        super(VoxelMorph, self).__init__(**kwargs)

        self.unet = Unet(
            inshape=inshape, infeats=unet_in_channel, nb_features=unet_channel_cfg
        )

        ndim = len(inshape)
        if ndim == 2:
            self.output2flow = nn.Conv2d(
                unet_channel_cfg[-1][-1], ndim, kernel_size=3, padding=1
            )
        elif ndim == 3:
            self.output2flow = nn.Conv3d(
                unet_channel_cfg[-1][-1], ndim, kernel_size=3, padding=1
            )
        else:
            raise NotImplementedError

        self.spatial_transformer = SpatialTransformer(size=inshape)

        self.lambda_grad = lambda_grad

        if loss_cfg is None:
            loss_cfg = dict(type="MSE")
        self.loss = VOXELMORPH_LOSS.build(loss_cfg)

    def forward(self, img, gt_seg_map, mode, **kwargs):
        moving_img = img[:, 0, :, :].unsqueeze(1)
        fixed_img = img[:, 1, :, :].unsqueeze(1)

        flow = self.output2flow(self.unet(img))
        wraped_img = self.spatial_transformer(moving_img, flow)

        if mode == "loss":
            loss_voxelmorph = self.loss.loss(wraped_img, fixed_img)

            grad_loss = Grad("l2").loss(None, flow)

            # model train step will sum up all the losses
            # with "loss" in name
            return {
                "loss_vm": loss_voxelmorph,
                "loss_grad_lambda": grad_loss * self.lambda_grad,
                "L_vm": loss_voxelmorph,
                "L_grad": grad_loss,
            }

        elif mode == "predict":
            # warped_img = einops.rearrange(warped_img, "b c h w -> b h w c")
            # flow = einops.rearrange(flow, "b c h w -> b h w c")
            # {"warped_img": warped_img, "flow": flow}
            data_sample = BaseDataElement(wraped_img=wraped_img, flow=flow)
            # NOTE(YL 07/21):: I don't get it. The evaluator requires
            # the output to be sequence of data samples. Should I
            # decouple the batch dimension?
            return [data_sample]

        else:
            raise NotImplementedError


def dilate_seg_map(seg_map, win_size: int = 5, ndim: int = 2):
    """Dilate the segmentation map to avoid the boundary effect

    Args:
        seg_map (torch.Tensor): segmentation map
        win_size (int, optional): dilation window size. Defaults to 5.
        ndim (int, optional): number of dimensions. Defaults to 2.

    Returns:
        torch.Tensor: dilated segmentation map
    """
    padding = win_size // 2
    if ndim == 2:
        kernel = torch.ones((1, 1, win_size, win_size))
        kernel = kernel.to(seg_map.device).to(seg_map.dtype)
        seg_map = F.conv2d(seg_map, kernel, padding=padding)
    elif ndim == 3:
        kernel = torch.ones((1, 1, win_size, win_size, win_size))
        kernel = kernel.to(seg_map.device).to(seg_map.dtype)
        seg_map = F.conv3d(seg_map, kernel, padding=padding)
    else:
        raise NotImplementedError

    seg_map[seg_map > 0] = 1
    return seg_map


@MODELS.register_module()
class VoxelMorphSegMask(BaseModel):
    def __init__(
        self,
        inshape: tuple = (32, 32),
        unet_in_channel: int = 2,
        unet_channel_cfg=(
            (16, 32, 32, 32),  # encoder features
            (32, 32, 32, 32, 32, 16, 16),  # decoder features
        ),
        lambda_grad: float = 0.1,
        loss_cfg: tp.Optional[dict] = None,
        use_seg_mask: bool = True,
        **kwargs
    ):
        super(VoxelMorphSegMask, self).__init__(**kwargs)

        self.unet = Unet(
            inshape=inshape, infeats=unet_in_channel, nb_features=unet_channel_cfg
        )

        ndim = len(inshape)
        if ndim == 2:
            self.output2flow = nn.Conv2d(
                unet_channel_cfg[-1][-1], ndim, kernel_size=3, padding=1
            )
        elif ndim == 3:
            self.output2flow = nn.Conv3d(
                unet_channel_cfg[-1][-1], ndim, kernel_size=3, padding=1
            )
        else:
            raise NotImplementedError

        self.spatial_transformer = SpatialTransformer(size=inshape)
        self.use_seg_mask = use_seg_mask

        self.lambda_grad = lambda_grad

        if loss_cfg is None:
            loss_cfg = dict(type="MSE", return_loss_map=True)
        self.loss = VOXELMORPH_LOSS.build(loss_cfg)

    def forward(self, img, gt_seg_map, mode, **kwargs):
        moving_img = img[:, 0, :, :].unsqueeze(1)
        fixed_img = img[:, 1, :, :].unsqueeze(1)
        bs = moving_img.shape[0]

        roi = (gt_seg_map >= 1).to(gt_seg_map.dtype)
        roi = dilate_seg_map(roi.double(), win_size=5, ndim=3)
        roi = (roi >= 1).to(gt_seg_map.dtype)

        flow = self.output2flow(self.unet(img))
        wraped_img = self.spatial_transformer(moving_img, flow)

        if mode == "loss":
            loss_map, is_valid = self.loss.loss(wraped_img, fixed_img)

            if self.use_seg_mask:
                is_valid = torch.logical_and(is_valid, roi)

            loss_map[~is_valid] = 0
            batch_loss = loss_map.view(bs, -1).sum(dim=1) / is_valid.view(bs, -1).sum(
                dim=1
            )
            loss_voxelmorph = batch_loss.nanmean()

            grad_loss = Grad("l2").loss(None, flow)

            # model train step will sum up all the losses
            # with "loss" in name
            return {
                "loss_vm": loss_voxelmorph,
                "loss_grad_lambda": grad_loss * self.lambda_grad,
                "L_vm": loss_voxelmorph,
                "L_grad": grad_loss,
            }

        elif mode == "predict":
            # warped_img = einops.rearrange(warped_img, "b c h w -> b h w c")
            # flow = einops.rearrange(flow, "b c h w -> b h w c")
            # {"warped_img": warped_img, "flow": flow}
            data_sample = BaseDataElement(wraped_img=wraped_img, flow=flow, roi_map=roi)
            # NOTE(YL 07/21):: I don't get it. The evaluator requires
            # the output to be sequence of data samples. Should I
            # decouple the batch dimension?
            return [data_sample]

        else:
            raise NotImplementedError
