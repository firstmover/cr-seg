#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : visualization.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 05/01/2023
#
# This file is part of cr_seg

"""

"""

import glob
from os import path as osp

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance
from tqdm import tqdm

import nibabel as nib

from .memory import memory
from .metric import compute_dice


def compute_seg_map_metric_sequence(pred_seg_map_file_list: list[str]):
    pred_seg_map_list = [nib.load(p).get_fdata() for p in tqdm(pred_seg_map_file_list)]

    dice_list = []
    hd_list = []
    hd95_list = []
    for i in range(len(pred_seg_map_list) - 1):
        pred_seg_map_1 = torch.tensor(pred_seg_map_list[i])
        pred_seg_map_2 = torch.tensor(pred_seg_map_list[i + 1])

        dice = compute_dice(pred_seg_map_1, pred_seg_map_2)
        dice_list.append(dice)

        # first dim: batch, second dim: num class
        pred_seg_map_1_unsqueeze = pred_seg_map_1.unsqueeze(0).unsqueeze(0)
        pred_seg_map_2_unsqueeze = pred_seg_map_2.unsqueeze(0).unsqueeze(0)
        hd = compute_hausdorff_distance(
            pred_seg_map_1_unsqueeze, pred_seg_map_2_unsqueeze
        )
        hd_list.append(hd.item())

        hd95 = compute_hausdorff_distance(
            pred_seg_map_1_unsqueeze, pred_seg_map_2_unsqueeze, percentile=95
        )
        hd95_list.append(hd95.item())

    return dice_list, hd_list, hd95_list


@memory.cache
def compute_seg_map_metric_sequence_from_dir(pred_seg_map_dir) -> np.ndarray:
    pred_seg_map_file_list = glob.glob(osp.join(pred_seg_map_dir, "*.nii.gz"))
    pred_seg_map_file_list.sort()
    dice_list, hd_list, hd95_list = compute_seg_map_metric_sequence(
        pred_seg_map_file_list
    )
    dice_list = np.array(dice_list)
    hd_list = np.array(hd_list)
    hd95_list = np.array(hd95_list)
    return dice_list, hd_list, hd95_list
