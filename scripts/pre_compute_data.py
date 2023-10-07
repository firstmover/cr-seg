#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : pre_compute_data.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/06/2023
#
# This file is part of med_img_ssl

"""

"""
import os
from os import path as osp

from cr_seg.datasets.placenta.raw import prepare_3d_placenta_segmentation_dataset_raw
from cr_seg.datasets.placenta.utils import _get_placenta_unlabelled_data_img_shape_list


def main():
    data_root = os.getenv("DATA_ROOT")
    if data_root is None or not osp.exists(data_root):
        raise ValueError("Please set DATA_ROOT in your environment variable.")
    data_root = osp.join(data_root, "4D")
    _get_placenta_unlabelled_data_img_shape_list(data_root)

    prepare_3d_placenta_segmentation_dataset_raw()

    prepare_3d_placenta_segmentation_dataset_raw(
        corrected_label_tag="split-nifti-raw-revised-2023-06-06"
    )


if __name__ == "__main__":
    main()
