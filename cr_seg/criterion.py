#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : criterion.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 01/25/2023
#
# This file is part of cr_seg

"""

"""


from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss

from mmengine.registry import MODELS

LOSSES = MODELS


LOSSES.register_module(module=DiceCELoss, name="DiceCELoss")
LOSSES.register_module(module=DiceFocalLoss, name="DiceFocalLoss")
LOSSES.register_module(module=DiceLoss, name="DiceLoss")
LOSSES.register_module(module=FocalLoss, name="FocalLoss")


def build_loss(cfg):
    return LOSSES.build(cfg)
