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

import logging
import typing as tp
from functools import partial
from os import path as osp

import torch

from mmengine.dataset.base_dataset import Compose as MMCompose
from mmengine.registry import DATASETS

from .dataset import (
    Placenta3DDatasetRaw,
    Placenta3DPairedUnlabelledDataset,
    Placenta3DUnlabelledDataset,
)
from .raw import prepare_3d_placenta_segmentation_dataset_raw
from .utils import _filename2subj_name_frame_idx

logger = logging.getLogger(__name__)


@DATASETS.register_module(
    name="placenta_3d_trimed_normed_cross_validation_for_cr", force=False
)
def build_placenta_3d_dataset_cross_validation_for_cr(
    train: bool = True,
    n_repeat_train: int = 32,
    norm_method: str = "linear_0_99_percentile",
    transform=None,
    num_fold: int = 5,
    val_fold_idx: int = 0,
    include_unlabeled: bool = False,
    use_raw: bool = False,
    corrected_label_tag: tp.Optional[str] = None,
):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = MMCompose(transform)

    _dataset_create_func = partial(
        Placenta3DDatasetRaw,
        transform=transform,
        norm_method=norm_method,
        corrected_label_tag=corrected_label_tag,
    )
    filename_list = prepare_3d_placenta_segmentation_dataset_raw()[2]

    # get all subj name list
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
        # labeled
        subj_name_list = [unique_subj_name_list[i] for i in train_subj_idx_list]
        dataset = _dataset_create_func(
            subj_name_list=subj_name_list,
        )
        dataset = torch.utils.data.ConcatDataset([dataset] * n_repeat_train)

        # unlabeled
        if include_unlabeled:
            unlabeled_dataset = Placenta3DUnlabelledDataset(
                norm_method=norm_method,
                transform=transform,
                subj_name_list=subj_name_list,
            )
            unlabeled_dataset = torch.utils.data.ConcatDataset(
                [unlabeled_dataset] * n_repeat_train
            )

            dataset = torch.utils.data.ConcatDataset([dataset, unlabeled_dataset])

        return dataset

    else:
        # val only return labeled and never repeat the dataset
        val_subj_name_list = [unique_subj_name_list[i] for i in val_subj_idx_list]
        val_file_idx_list = []
        train_file_idx_list = []
        for i, subj_name in enumerate(subj_name_list):
            if subj_name in val_subj_name_list:
                val_file_idx_list.append(i)
            else:
                train_file_idx_list.append(i)
        split_tag2filename_list = {
            "train": [filename_list[i] for i in train_file_idx_list],
            "val": [filename_list[i] for i in val_file_idx_list],
        }
        dataset = _dataset_create_func(
            split_tag2filename_list=split_tag2filename_list,
        )
        return dataset


@DATASETS.register_module(
    name="placenta_3d_trimed_normed_cross_validation_for_cr_with_tc", force=False
)
def build_placenta_3d_dataset_cross_validation_for_cr_with_tc(
    train: bool = True,
    n_repeat_train: int = 32,
    norm_method: str = "linear_0_99_percentile",
    transform=None,
    num_fold: int = 5,
    val_fold_idx: int = 0,
    include_unlabeled: bool = True,
    use_raw: bool = False,
    corrected_label_tag: tp.Optional[str] = None,
):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = MMCompose(transform)

    assert include_unlabeled, (
        "If not include unlabeled data, use "
        "placenta_3d_trimed_normed_cross_validation_for_cr instead."
    )

    _dataset_create_func = partial(
        Placenta3DDatasetRaw,
        transform=transform,
        norm_method=norm_method,
        corrected_label_tag=corrected_label_tag,
    )
    filename_list = prepare_3d_placenta_segmentation_dataset_raw()[2]

    # get all subj name list
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
        # labeled
        subj_name_list = [unique_subj_name_list[i] for i in train_subj_idx_list]
        dataset = _dataset_create_func(
            subj_name_list=subj_name_list,
        )
        dataset = torch.utils.data.ConcatDataset([dataset] * n_repeat_train)

        # unlabeled
        if include_unlabeled:
            unlabeled_dataset_t = Placenta3DPairedUnlabelledDataset(
                norm_method=norm_method,
                transform=transform,
                subj_name_list=subj_name_list,
                mode="t",
            )
            unlabeled_dataset_t = torch.utils.data.ConcatDataset(
                [unlabeled_dataset_t] * n_repeat_train
            )

            unlabeled_dataset_t_plus_1 = Placenta3DPairedUnlabelledDataset(
                norm_method=norm_method,
                transform=transform,
                subj_name_list=subj_name_list,
                mode="t+1",
            )
            unlabeled_dataset_t_plus_1 = torch.utils.data.ConcatDataset(
                [unlabeled_dataset_t_plus_1] * n_repeat_train
            )

            dataset = torch.utils.data.ConcatDataset(
                [dataset, unlabeled_dataset_t, unlabeled_dataset_t_plus_1]
            )

        return dataset

    else:
        # val only return labeled and never repeat the dataset
        val_subj_name_list = [unique_subj_name_list[i] for i in val_subj_idx_list]
        val_file_idx_list = []
        train_file_idx_list = []
        for i, subj_name in enumerate(subj_name_list):
            if subj_name in val_subj_name_list:
                val_file_idx_list.append(i)
            else:
                train_file_idx_list.append(i)
        split_tag2filename_list = {
            "train": [filename_list[i] for i in train_file_idx_list],
            "val": [filename_list[i] for i in val_file_idx_list],
        }
        dataset = _dataset_create_func(
            split_tag2filename_list=split_tag2filename_list,
        )
        return dataset
