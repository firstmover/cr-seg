#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 06/29/2023
#
# Distributed under terms of the MIT license.

"""

"""

from os import path as osp

from mmengine.dataset.base_dataset import Compose as MMCompose
from mmengine.registry import DATASETS

from .dataset import Placenta3DUnlabelledDataset
from .raw import prepare_3d_placenta_segmentation_dataset_raw
from .utils import _filename2subj_name_frame_idx


# mainly for inference time series results
@DATASETS.register_module(
    name="placenta_3d_trimed_normed_cross_validation_unlabeled", force=False
)
def build_placenta_3d_dataset_cross_validation_unlabeled(
    train: bool = True,
    norm_method: str = "linear_0_99_percentile",
    transform=None,
    num_fold: int = 5,
    val_fold_idx: int = 0,
):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = MMCompose(transform)

    filename_list = prepare_3d_placenta_segmentation_dataset_raw()[2]

    subj_name_list = [
        _filename2subj_name_frame_idx(osp.basename(p))[0] for p in filename_list
    ]
    unique_subj_name_list = list(sorted(set(subj_name_list)))
    len_subj = len(unique_subj_name_list)

    fold_idx_list: list[list[int]] = [
        list(range(i, len_subj, num_fold)) for i in range(num_fold)
    ]

    train_subj_idx_list = []
    val_subj_idx_list = []
    for i, f_idx_list in enumerate(fold_idx_list):
        if i == val_fold_idx:
            val_subj_idx_list = f_idx_list
        else:
            train_subj_idx_list += f_idx_list

    if train:
        subj_name_list = [unique_subj_name_list[i] for i in train_subj_idx_list]
    else:
        subj_name_list = [unique_subj_name_list[i] for i in val_subj_idx_list]

    unlabeled_dataset = Placenta3DUnlabelledDataset(
        norm_method=norm_method,
        transform=transform,
        subj_name_list=subj_name_list,
    )

    return unlabeled_dataset
