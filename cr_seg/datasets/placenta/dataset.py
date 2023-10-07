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
import typing as tp
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from monai.transforms import LoadImage

from .raw import prepare_3d_placenta_segmentation_dataset_raw
from .utils import (
    _file_name_is_in_subj_name_list,
    _filename2subj_name_frame_idx,
    _get_placenta_unlabelled_data_img_shape_list,
    _get_placenta_unlabelled_data_path,
    _normalize_image,
)

logger = logging.getLogger(__name__)


class Placenta3DUnlabelledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        norm_method: str = "linear_0_256",
        transform=None,
        subj_name_list=None,
        min_size=32,
    ):
        data_root = os.getenv("DATA_ROOT")
        if data_root is None or not osp.exists(data_root):
            raise ValueError("Please set DATA_ROOT in your environment variable.")
        data_root = osp.join(data_root, "4D")
        image_path_list = _get_placenta_unlabelled_data_path(data_root)
        img_shape_list = _get_placenta_unlabelled_data_img_shape_list(data_root)
        is_valid_img = np.all(img_shape_list >= min_size, axis=1)
        self.image_path_list = [i for i, j in zip(image_path_list, is_valid_img) if j]

        if subj_name_list is not None:
            image_subj_name_list = [
                _filename2subj_name_frame_idx(osp.basename(path))[0]
                for path in self.image_path_list
            ]  # subj name is already sorted
            subj_name_list = sorted(subj_name_list)

            set_image_subj_name_list = set(image_subj_name_list)
            missing_subj_name_list = [
                s for s in subj_name_list if s not in set_image_subj_name_list
            ]
            if len(missing_subj_name_list) > 0:
                raise ValueError(
                    f"Missing subj name in image_path_list: {missing_subj_name_list}. "
                    "You might be passing a wrong subj name. "
                )

            idx_sel_list = []
            for i, i_subj_name in enumerate(image_subj_name_list):
                if i_subj_name in subj_name_list:
                    idx_sel_list.append(i)

            self.image_path_list = [self.image_path_list[i] for i in idx_sel_list]

        self.norm_method = norm_method

        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]

        filename = osp.basename(image_path)

        image = LoadImage(image_only=True)(image_path).numpy()
        assert len(image.shape) == 3
        image = _normalize_image(norm_method=self.norm_method, image=image)
        image = image[:, :, :, np.newaxis]

        # a placeholder gt_seg_map
        gt_seg_map = np.zeros_like(image)[..., 0].astype(np.uint8)

        data_dict = {"img": image, "filename": filename, "gt_seg_map": gt_seg_map}
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class Placenta3DPairedUnlabelledDataset(Placenta3DUnlabelledDataset):
    def __init__(
        self,
        norm_method: str = "linear_0_256",
        transform=None,
        subj_name_list=None,
        min_size=32,
        mode: str = "t",  # 't' or 't+1'
    ):
        super().__init__(
            norm_method=norm_method,
            transform=transform,
            subj_name_list=subj_name_list,
            min_size=min_size,
        )

        self.ret_sample_name = mode
        assert self.ret_sample_name in ["t", "t+1"], self.ret_sample_name

        # prepare the index of t and t+1 samples

        subj_name_frame_idx_list = [
            _filename2subj_name_frame_idx(osp.basename(path))
            for path in self.image_path_list
        ]
        subj_name2frame_idx_list = defaultdict(list)
        for subj_name, frame_idx in subj_name_frame_idx_list:
            subj_name2frame_idx_list[subj_name].append(frame_idx)

        subj_name2min_max_frame_idx = {}
        for subj_name, frame_idx_list in subj_name2frame_idx_list.items():
            subj_name2min_max_frame_idx[subj_name] = (
                min(frame_idx_list),
                max(frame_idx_list),
            )

        # for t index, we remove the last frame
        # for t+1 index, we remove the first frame
        idx_t_list = []
        idx_t_plus_1_list = []
        for i, sname_fidx in enumerate(subj_name_frame_idx_list):
            subj_name, frame_idx = sname_fidx
            min_frame_idx, max_frame_idx = subj_name2min_max_frame_idx[subj_name]
            if frame_idx == min_frame_idx:
                idx_t_list.append(i)
            elif frame_idx == max_frame_idx:
                idx_t_plus_1_list.append(i)
            else:
                idx_t_list.append(i)
                idx_t_plus_1_list.append(i)

        assert len(idx_t_list) == len(idx_t_plus_1_list)

        img_path_t_list = [self.image_path_list[i] for i in idx_t_list]
        img_path_t_plus_1_list = [self.image_path_list[i] for i in idx_t_plus_1_list]

        # check if the t and t+1 images are paired
        for _i, (path_t, path_t_plus_1) in enumerate(
            zip(img_path_t_list, img_path_t_plus_1_list)
        ):
            subj_name_t, frame_idx_t = _filename2subj_name_frame_idx(
                osp.basename(path_t)
            )
            subj_name_t_plus_1, frame_idx_t_plus_1 = _filename2subj_name_frame_idx(
                osp.basename(path_t_plus_1)
            )
            assert subj_name_t == subj_name_t_plus_1
            assert frame_idx_t + 1 == frame_idx_t_plus_1

        if self.ret_sample_name == "t":
            self.image_path_list = img_path_t_list
        elif self.ret_sample_name == "t+1":
            self.image_path_list = img_path_t_plus_1_list
        else:
            raise ValueError(self.ret_sample_name)


class Placenta3DDatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        norm_method: str = "linear_0_99_percentile",
        transform=None,
        subj_name_list=None,
        include_filename_list=None,
        split_tag2filename_list: tp.Optional[dict[str, list]] = None,
        corrected_label_tag: tp.Optional[str] = None,
    ):
        (
            self.image_array,
            self.label_array,
            self.filename_list,
        ) = prepare_3d_placenta_segmentation_dataset_raw(
            norm_method, corrected_label_tag
        )

        if subj_name_list is not None:
            idx_sel_list = [
                i
                for i, f in enumerate(self.filename_list)
                if _file_name_is_in_subj_name_list(f, subj_name_list)
            ]

            self.image_array = [self.image_array[i] for i in idx_sel_list]
            self.label_array = [self.label_array[i] for i in idx_sel_list]
            self.filename_list = [self.filename_list[i] for i in idx_sel_list]

        if include_filename_list is not None:
            idx_sel_list = [
                i
                for i, f in enumerate(self.filename_list)
                if f in include_filename_list
            ]

            self.image_array = [self.image_array[i] for i in idx_sel_list]
            self.label_array = [self.label_array[i] for i in idx_sel_list]
            self.filename_list = [self.filename_list[i] for i in idx_sel_list]

        self.filename2tag = None
        if split_tag2filename_list is not None:
            if subj_name_list is not None:
                raise ValueError(
                    "subj_name_list and split_dict cannot be both specified"
                )
            assert all(
                [k in ["train", "val", "test"] for k in split_tag2filename_list.keys()]
            )

            self.filename2tag = {}
            for split_tag, filename_list in split_tag2filename_list.items():
                self.filename2tag.update({f: split_tag for f in filename_list})

        self.transform = transform

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        label = self.label_array[idx]
        filename = self.filename_list[idx]

        image = image[:, :, :, np.newaxis]

        data_dict = {"img": image, "gt_seg_map": label, "filename": filename}
        if self.filename2tag is not None:
            data_dict["split_tag"] = self.filename2tag[filename]

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class Placenta3DUnlabelledRegiDataset(Placenta3DUnlabelledDataset):
    def __init__(
        self,
        norm_method: str = "linear_0_99_percentile",
        transform=None,
        subj_name_list=None,
        split_tag2subj_name_list: tp.Optional[dict[str, list]] = None,
        pairing_pattern: str = "all",
    ):
        super().__init__(
            norm_method=norm_method,
            transform=None,
            subj_name_list=subj_name_list,
        )

        # group image by subjects
        filename_list = [osp.basename(f) for f in self.image_path_list]
        subj_name_frame_idx_list: list[tuple[str, int]] = [
            _filename2subj_name_frame_idx(f) for f in filename_list
        ]
        subj2f_idx_img_idx_subj = defaultdict(list)
        for image_idx, (subj_name, frame_idx) in enumerate(subj_name_frame_idx_list):
            subj2f_idx_img_idx_subj[subj_name].append((frame_idx, image_idx, subj_name))

        # make each registration sample a pair of images from the same subject
        self.pair_list: list[tuple[tuple[int, int, str], tuple[int, int, str]]] = []
        for _subj_name, f_idx_img_idx_subj_list in subj2f_idx_img_idx_subj.items():
            # sorted by frame_idx
            f_idx_img_idx_subj_list = sorted(
                f_idx_img_idx_subj_list, key=lambda x: x[0]
            )
            len_frame = len(f_idx_img_idx_subj_list)

            # each frame is paired with all other frames
            if pairing_pattern == "all":
                for i in range(len_frame):
                    for j in range(len_frame):
                        if i == j:
                            continue
                        self.pair_list.append(
                            (f_idx_img_idx_subj_list[i], f_idx_img_idx_subj_list[j])
                        )

            # each frame is paired with the next frame
            elif pairing_pattern == "consecutive":
                for i in range(len_frame - 1):
                    self.pair_list.append(
                        (f_idx_img_idx_subj_list[i], f_idx_img_idx_subj_list[i + 1])
                    )

            else:
                raise ValueError(f"Unknown pairing_pattern: {pairing_pattern}")

        self.subj_name2tag = None
        if split_tag2subj_name_list is not None:
            assert all(
                [k in ["train", "val", "test"] for k in split_tag2subj_name_list.keys()]
            )

            self.subj_name2tag = {}
            for split_tag, subj_name_list in split_tag2subj_name_list.items():
                self.subj_name2tag.update({f: split_tag for f in subj_name_list})

        self.transform = transform

    def __len__(self):
        return len(self.pair_list)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = LoadImage(image_only=True)(image_path).numpy()
        assert len(image.shape) == 3
        image = _normalize_image(norm_method=self.norm_method, image=image)
        image = image[:, :, :, np.newaxis]
        return image

    def __getitem__(self, idx):
        pair1, pair2 = self.pair_list[idx]
        _f_idx1, i_idx1, subj1 = pair1
        _f_idx2, i_idx2, subj2 = pair2
        assert subj1 == subj2

        image1 = self._load_image(self.image_path_list[i_idx1])
        image2 = self._load_image(self.image_path_list[i_idx2])
        image_pair = np.concatenate([image1, image2], axis=-1)

        # placeholder gt_seg_map
        gt_seg_map1 = np.zeros_like(image1).astype(np.uint8)[..., 0]
        # gt_seg_map2 = np.zeros_like(image2).astype(np.uint8)
        # gt_seg_map_pair = np.concatenate([gt_seg_map1, gt_seg_map2], axis=-1)

        data_dict = {
            "img": image_pair,
            "gt_seg_map": gt_seg_map1,
            "subj_name": subj1,
            "frame_idx_pair": (_f_idx1, _f_idx2),
        }
        if self.subj_name2tag is not None:
            data_dict["split_tag"] = self.subj_name2tag[subj1]

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
