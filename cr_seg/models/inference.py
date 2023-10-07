#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 07/28/2023
#
# Distributed under terms of the MIT license.

"""Inference models

To make things simple, I am going to define inference model
for each of the model in the model zoo. The inference model
will be a wrapper of the original model.
"""


import torch

from mmengine import Registry
from mmengine.registry import MODELS

from .unet_3d_nobatchnorm import UNet3dDropout


@MODELS.register_module()
class UNet3dDropoutInference(UNet3dDropout):
    def __init__(self, inference_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_cfg = inference_cfg

        # TODO(YL 07/28):: create a inference distributor and gather
        self.handler = INFERENCE_DATA_HANDLER.build(inference_cfg)

    def forward(self, img, gt_seg_map, mode):
        assert mode in [
            "predict",
            "predict_and_logits",
        ], "Inference model only support predict mode"

        img_patches = self.handler.distribute(img)
        _, logits_patches = super().forward(
            img_patches, None, mode="predict_and_logits"
        )  # type: ignore
        logits = self.handler.gather(logits_patches, img.shape)

        preds = torch.argmax(logits, dim=1)

        if mode == "predict":
            return preds
        elif mode == "predict_and_logits":
            return preds, logits


INFERENCE_DATA_HANDLER = Registry(
    "inference_data_handler",
    scope="mmengine",
    locations=["cr_seg.models.inference"],
)


@INFERENCE_DATA_HANDLER.register_module()
class AllCrop:
    def distribute(self, img: torch.Tensor) -> torch.Tensor:
        """Return the same image"""
        return img

    def gather(self, img_patches: torch.Tensor, img_shape: tuple) -> torch.Tensor:
        """Return the same image"""
        return img_patches
