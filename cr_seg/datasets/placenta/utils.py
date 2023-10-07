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
import os
from os import path as osp

import numpy as np
from joblib import Parallel, delayed
from monai.transforms import LoadImage

from cr_seg.memory import memory

logger = logging.getLogger(__name__)


def _normalize_image(norm_method, image):
    if norm_method == "linear_0_256":
        # image scale from 0 - 256 to 0 - 1, clip the rest to 1 and 0
        image = image / 256
        image[image > 1] = 1
        image[image < 0] = 0

    elif norm_method == "linear_0_95_percentile":
        # image scale from 0 - 95 percentile to 0 - 1, clip the rest to 1 and 0
        image = image / np.percentile(image, 95)
        image[image > 1] = 1
        image[image < 0] = 0

    elif norm_method == "linear_0_99_percentile":
        # image scale from 0 - 99 percentile to 0 - 1, clip the rest to 1 and 0
        image = image / np.percentile(image, 99)
        image[image > 1] = 1
        image[image < 0] = 0

    elif norm_method == "linear_1_percentile_99_percentile":
        # image scale from 1 percentile - 99 percentile to 0 - 1,
        # clip the rest to 1 and 0
        image = image - np.percentile(image, 1)
        image = image / (np.percentile(image, 99) - np.percentile(image, 1))
        image[image > 1] = 1
        image[image < 0] = 0

    else:
        raise ValueError(f"norm_method {norm_method} not supported")

    return image


def _file_name_is_in_subj_name_list(file_name, subj_name_list):
    return any(["/" + subj_name + "/" in file_name for subj_name in subj_name_list])


def _get_placenta_unlabelled_data_path(data_root):
    subj_file_name_list = sorted(os.listdir(data_root))

    # find all files in data_root recursively
    # and get file path that contains "raw_3D" and file name ends with "nii.gz"

    image_path_list = []
    for subj_name in subj_file_name_list:
        img_dir_path = osp.join(data_root, subj_name, "raw_3D")
        img_file_name_list = sorted(os.listdir(img_dir_path))
        for img_file_name in img_file_name_list:
            if img_file_name.endswith("nii.gz"):
                image_path_list.append(osp.join(img_dir_path, img_file_name))

    return image_path_list


def _filename2subj_name_frame_idx(filename: str) -> tuple[str, int]:
    subj_name = filename.split("_")[0]
    frame_idx = int(filename.split("_")[1].split(".")[0])
    return subj_name, frame_idx


@memory.cache
def _get_placenta_unlabelled_data_img_shape_list(data_root):
    image_path_list = _get_placenta_unlabelled_data_path(data_root)

    def _get_shape(path):
        return LoadImage(image_only=True)(path).shape

    img_shape_list = Parallel(n_jobs=8)(
        delayed(_get_shape)(path) for path in image_path_list
    )
    img_shape_list = np.array(img_shape_list)
    return img_shape_list
