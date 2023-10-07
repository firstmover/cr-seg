#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 07/23/2023
#
# Distributed under terms of the MIT license.

"""

"""

import os
import typing as tp
from functools import partial
from os import path as osp

import torch

from mmengine.dataset.base_dataset import Compose as MMCompose
from mmengine.registry import DATASETS

from .dataset import Placenta3DUnlabelledRegiDataset
from .raw import prepare_3d_placenta_segmentation_dataset_raw
from .utils import _filename2subj_name_frame_idx, _get_placenta_unlabelled_data_path


def _get_subj_name_list_in_unlabeled_data():
    data_root = os.getenv("DATA_ROOT")
    if data_root is None or not osp.exists(data_root):
        raise ValueError("Please set DATA_ROOT in your environment variable.")
    data_root = osp.join(data_root, "4D")
    filename_list = _get_placenta_unlabelled_data_path(data_root)
    subj_name_list = [
        _filename2subj_name_frame_idx(osp.basename(p))[0] for p in filename_list
    ]
    unique_subj_name_list = list(sorted(set(subj_name_list)))
    len_subj = len(unique_subj_name_list)
    return len_subj, unique_subj_name_list


@DATASETS.register_module(name="placenta_3d_registration", force=False)
def build_placenta_3d_registration(
    train: bool = True,
    n_repeat_train: int = 1,
    norm_method: str = "linear_0_99_percentile",
    transform=None,
    num_fold: int = 5,
    val_fold_idx: int = 0,
    use_raw: bool = True,
    corrected_label_tag: tp.Optional[str] = None,
    validation_sub_sample_rate: float = 1.0,
    pairing_pattern: str = "consecutive",
):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = MMCompose(transform)

    if not use_raw:
        raise NotImplementedError

    # we decide train and validation split based on raw labeled data
    # not the unlabelled data to be consistent

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
    val_subj_name_list = [unique_subj_name_list[i] for i in val_subj_idx_list]
    train_subj_name_list = [unique_subj_name_list[i] for i in train_subj_idx_list]

    # we keep only the subjects that are in the unlabeled data
    _, unique_subj_name_list_in_unlabeled = _get_subj_name_list_in_unlabeled_data()
    val_subj_name_list = [
        s for s in val_subj_name_list if s in unique_subj_name_list_in_unlabeled
    ]
    train_subj_name_list = [
        s for s in train_subj_name_list if s in unique_subj_name_list_in_unlabeled
    ]

    _dataset_create_func = partial(
        Placenta3DUnlabelledRegiDataset,
        transform=transform,
        norm_method=norm_method,
        pairing_pattern=pairing_pattern,
    )

    if train:
        dataset = _dataset_create_func(subj_name_list=train_subj_name_list)
        dataset = torch.utils.data.ConcatDataset([dataset] * n_repeat_train)
        return dataset

    else:
        # val only return labeled and never repeat the dataset
        split_tag2subj = {
            "train": train_subj_name_list,
            "val": val_subj_name_list,
        }
        dataset = _dataset_create_func(
            subj_name_list=val_subj_name_list + train_subj_name_list,
            split_tag2subj_name_list=split_tag2subj,
        )

        if validation_sub_sample_rate < 1.0:
            step = int(1.0 / validation_sub_sample_rate)
            indices = list(range(0, len(dataset), step))
            dataset = torch.utils.data.Subset(
                dataset,
                indices=indices,
            )

        return dataset
