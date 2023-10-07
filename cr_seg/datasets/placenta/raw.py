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

import glob
import logging
import os
import typing as tp
from os import path as osp

from monai.data import Dataset
from monai.transforms import Compose, LoadImaged
from tqdm import tqdm

from cr_seg.memory import memory

from .utils import _normalize_image

logger = logging.getLogger(__name__)


@memory.cache
def prepare_3d_placenta_segmentation_dataset_raw(
    norm_method: str = "linear_0_99_percentile",
    corrected_label_tag: tp.Optional[str] = None,
):
    # prepare data
    data_root = os.getenv("DATA_ROOT")
    if data_root is None or not osp.exists(data_root):
        raise ValueError("Please set DATA_ROOT in your environment variable.")
    data_root = osp.join(data_root, "split-nifti-raw")

    def _orignal_label_path_to_corrected_path_if_exist(original_path: str):
        assert "split-nifti-raw" in original_path and corrected_label_tag is not None
        corrected_path = original_path.replace("split-nifti-raw", corrected_label_tag)
        if osp.exists(corrected_path):
            return corrected_path
        else:
            return original_path

    # image_path_list, label_path_list = get_placenta_data_path(data_root)
    all_dirs = glob.glob(osp.join(data_root, "*/"))
    image_path_list = []
    label_path_list = []
    for d in all_dirs:
        img_dir = os.path.join(d, "volume")
        label_dir = os.path.join(d, "segmentation")

        assert osp.exists(img_dir), f"{img_dir} does not exist"
        assert osp.exists(label_dir), f"{label_dir} does not exist"

        # NOTE(YL 01/13):: this could be too hacky
        img_file_list = list(sorted(glob.glob(os.path.join(img_dir, "*.nii*"))))
        label_file_list = list(sorted(glob.glob(os.path.join(label_dir, "*.nii*"))))
        assert len(img_file_list) == len(label_file_list), f"{d}"

        if corrected_label_tag is not None:
            label_file_list = [
                _orignal_label_path_to_corrected_path_if_exist(p)
                for p in label_file_list
            ]

        image_path_list.extend(img_file_list)
        label_path_list.extend(label_file_list)

    # NOTE(YL 06/04):: it seems that the shape of label
    # is different from the shape of image
    ignore_file_name_list = [
        "MAP-C303_0192_splitnum_0385",
        "MAP-C303_75_even_splitnum_0149",
    ]
    for img_path, lab_path in zip(image_path_list, label_path_list):
        for ignore_file_name in ignore_file_name_list:
            if ignore_file_name in img_path:
                image_path_list.remove(img_path)
                label_path_list.remove(lab_path)
                break

    data_dict = [
        {"image": image_path, "label": label_path}
        for image_path, label_path in zip(image_path_list, label_path_list)
    ]
    transforms = Compose([LoadImaged(keys=["image", "label"])])
    ds = Dataset(data=data_dict, transform=transforms)

    image_list = []
    label_list = []
    filename_list = []
    for data_dict in tqdm(ds):
        image = data_dict["image"][:, :, :, 0].numpy()

        label = data_dict["label"].numpy()
        if len(label.shape) == 4:
            assert label.shape[-1] == 1
            label = label[:, :, :, 0]
        else:
            assert len(label.shape) == 3

        if image.shape != label.shape:
            raise ValueError(f"image.shape {image.shape} != label.shape {label.shape}")

        image = _normalize_image(norm_method=norm_method, image=image)

        image_list.append(image)
        label_list.append(label)
        filename_list.append(data_dict["image_meta_dict"]["filename_or_obj"])

    assert len(image_list) == len(label_list) == len(filename_list)

    return image_list, label_list, filename_list
